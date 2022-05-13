from __future__ import print_function
import copy
import time
import numpy as np

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

from utils import AverageMeter, WeightEMA
from progress.bar import Bar as Bar

__all__ = ['classifier_warmup']


def classifier_warmup(model, train_labeled_set, train_unlabeled_set, N_SAMPLES_PER_CLASS, num_class, use_cuda, args):

    # Hypers used during warmup
    epochs = args.warm_tfe  # 10
    lr = args.lr_tfe  # 0.002
    ema_decay = args.ema_decay  # 0.999
    weight_decay = args.wd_tfe  # 5e-4
    batch_size = args.batch_size  # 64
    val_iteration = args.val_iteration  # 500

    # Construct dataloaders
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=batch_size,
                                          shuffle=False, num_workers=0, drop_last=False)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size,
                                            shuffle=False, num_workers=0, drop_last=False)

    tfe_model = weight_imprint(copy.deepcopy(model), train_labeled_set, num_class)

    # Fix the feature extractor and reinitialize the classifier
    for param in model.parameters():
        param.requires_grad = False
    model.output.reset_parameters()
    for param in model.output.parameters():
        param.requires_grad = True

    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    ema_optimizer = WeightEMA(model, ema_model, lr, alpha=ema_decay, wd=False)

    wd_params, non_wd_params = [], []
    for name, param in model.output.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias, conv2d.bias
        else:
            wd_params.append(param)
    param_list = [{'params': wd_params, 'weight_decay': weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.output.parameters()) / 1000000.0))

    optimizer = optim.Adam(param_list, lr=lr)

    # Generate TFE features in advance as the model and the data loaders are fixed anyway
    balanced_feature_set = TFE(labeled_trainloader, unlabeled_trainloader,
                               tfe_model, num_class, N_SAMPLES_PER_CLASS)
    balanced_feature_loader = data.DataLoader(balanced_feature_set, batch_size=batch_size,
                                              shuffle=True, num_workers=0, drop_last=True)

    # Main function
    for epoch in range(epochs):
        print('\ncRT: Epoch: [%d | %d] LR: %f' % (epoch + 1, epochs, optimizer.param_groups[0]['lr']))

        classifier_train(balanced_feature_loader, model, optimizer, None, ema_optimizer, val_iteration, use_cuda)

    return model, ema_model


def TFE(labeled_loader, unlabeled_loader, tfe_model, num_classes, num_samples_per_class):

    tfe_model.eval()
    with torch.no_grad():
        # ****************** extract features  ********************
        # extract features from labeled data
        for batch_idx, (inputs, targets, _) in enumerate(labeled_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            logits, _, features = tfe_model(inputs, return_feature=True)
            cls_probs = torch.softmax(logits, dim=1)
            features = features.squeeze()  # Note: a flatten is needed here
            if batch_idx == 0:
                labeled_feature_stack = features
                labeled_target_stack = targets
                labeled_cls_prob_stack = cls_probs
            else:
                labeled_feature_stack = torch.cat((labeled_feature_stack, features), 0)
                labeled_target_stack = torch.cat((labeled_target_stack, targets), 0)
                labeled_cls_prob_stack = torch.cat((labeled_cls_prob_stack, cls_probs), 0)
        # extract features from unlabeled data
        for batch_idx, (data_batch, _, _) in enumerate(unlabeled_loader):
            if hasattr(unlabeled_loader.dataset.transform, 'transform2'):  # FixMatch, ReMixMatch
                inputs_w, inputs_s, _ = data_batch
                inputs_s = inputs_s.cuda()
                inputs_w = inputs_w.cuda()

                _, _, features = tfe_model(inputs_s, return_feature=True)
                logits, _ = tfe_model(inputs_w)
            else:  # MixMatch
                inputs_w, _ = data_batch
                inputs_w = inputs_w.cuda()
                logits, _, features = tfe_model(inputs_w, return_feature=True)
            cls_probs = torch.softmax(logits, dim=1)
            _, targets = torch.max(cls_probs, dim=1)
            features = features.squeeze()
            if batch_idx == 0:
                unlabeled_feature_stack = features
                unlabeled_target_stack = targets
                unlabeled_cls_prob_stack = cls_probs
            else:
                unlabeled_feature_stack = torch.cat((unlabeled_feature_stack, features), 0)
                unlabeled_target_stack = torch.cat((unlabeled_target_stack, targets), 0)
                unlabeled_cls_prob_stack = torch.cat((unlabeled_cls_prob_stack, cls_probs), 0)

        # ****************** create TFE features for each class  ********************
        # create idx array for each class, per_cls_idx[i] contains all indices of images of class i
        labeled_set_idx = torch.tensor(list(range(len(labeled_feature_stack))))
        labeled_set_per_cls_idx = [labeled_set_idx[labeled_target_stack == i] for i in range(num_classes)]

        augment_features = []  # newly generated tfe features will be appended here
        augment_targets = []  # as well as their one-hot targets
        for cls_id in range(num_classes):
            if num_samples_per_class[cls_id] < max(num_samples_per_class):

                # how many we need for the cls
                augment_size = max(num_samples_per_class) - num_samples_per_class[cls_id]

                # create data belonging to class i
                current_cls_feats = labeled_feature_stack[labeled_target_stack == cls_id]

                # create data not belonging to class i
                other_labeled_data_idx = np.concatenate(labeled_set_per_cls_idx[:cls_id] + labeled_set_per_cls_idx[cls_id + 1:], axis=0)
                other_cls_feats = torch.cat([labeled_feature_stack[other_labeled_data_idx], unlabeled_feature_stack], dim=0)
                other_cls_probs = torch.cat([labeled_cls_prob_stack[other_labeled_data_idx], unlabeled_cls_prob_stack], dim=0)

                assert len(other_cls_feats) == len(other_cls_probs)
                # the total number of data should be the same for label-unlabel split, and current-the-rest split
                assert (len(other_cls_feats) + len(current_cls_feats)) == (len(labeled_feature_stack) + len(unlabeled_feature_stack))

                # sort other_cls_feats according to the probs assigned to class i
                probs4current_cls = other_cls_probs[:, cls_id]
                sorted_probs, order = probs4current_cls.sort(descending=True)  # sorted_probs = probs belonging to cls i
                other_cls_feats = other_cls_feats[order]

                # select features from the current class
                input_a_idx = np.random.choice(list(range(len(current_cls_feats))), augment_size, replace=True)
                # take first n features from all other classes
                input_b_idx = np.asarray(list(range(augment_size)))
                lambdas = np.random.beta(0.75, 0.75, size=augment_size)

                # do TFE
                for l, a_idx, b_idx in zip(lambdas, input_a_idx, input_b_idx):
                    tfe_input = l * current_cls_feats[a_idx] + (1 - l) * other_cls_feats[b_idx]  # [128]
                    tfe_target = torch.zeros((1, num_classes))
                    tfe_target[0, cls_id] = 1  # pseudo_label.tolist()
                    augment_features.append(tfe_input.view(1, -1))
                    augment_targets.append(tfe_target)

        # ****************** merge newly generated data with labeled dataset  ********************
        augment_features = torch.cat(augment_features, dim=0)
        augment_targets = torch.cat(augment_targets, dim=0).cuda()

        target_stack = torch.zeros(len(labeled_target_stack), num_classes).cuda().scatter_(1, labeled_target_stack.view(-1, 1), 1)
        new_feat_tensor = torch.cat([labeled_feature_stack, augment_features], dim=0)
        new_target_tensor = torch.cat([target_stack, augment_targets], dim=0)

    balanced_feature_set = data.dataset.TensorDataset(new_feat_tensor, new_target_tensor)
    return balanced_feature_set


def weight_imprint(model, labeled_set, num_classes):
    model = model.cuda()
    model.eval()

    labeledloader = data.DataLoader(labeled_set, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

    with torch.no_grad():
        bar = Bar('Classifier weight imprinting...', max=len(labeledloader))

        for batch_idx, (inputs, targets, _) in enumerate(labeledloader):
            inputs = inputs.cuda()
            _, _, features = model(inputs, True)
            output = features.squeeze()   # Note: a flatten is needed here

            if batch_idx == 0:
                output_stack = output
                target_stack = targets
            else:
                output_stack = torch.cat((output_stack, output), 0)
                target_stack = torch.cat((target_stack, targets), 0)

            bar.suffix = '({batch}/{size}'.format(batch=batch_idx + 1, size=len(labeledloader))
            bar.next()
        bar.finish()

    new_weight = torch.zeros(num_classes, model.output.in_features)

    for i in range(num_classes):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)

    model.output = torch.nn.Linear(model.output.in_features, num_classes, bias=False).cuda()
    model.output.weight.data = new_weight.cuda()

    model.eval()
    return model


def classifier_train(labeled_trainloader, model, optimizer, scheduler, ema_optimizer, val_iteration, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=val_iteration)
    labeled_train_iter = iter(labeled_trainloader)

    model.eval()
    for batch_idx in range(val_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        data_time.update(time.time() - end)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)  # targets are one-hot

        outputs = model.output(inputs_x)

        loss = (-F.log_softmax(outputs, dim=1) * targets_x).sum(dim=1)
        loss = loss.mean()
        acc = (torch.argmax(outputs, dim=1) == torch.argmax(targets_x, dim=1)).float().sum() / len(targets_x)

        # Record loss and acc
        losses.update(loss.item(), inputs_x.size(0))
        train_acc.update(acc.item(), inputs_x.size(0))

        # Compute gradient and apply SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Train_Acc: {train_acc:.4f}'.format(
                     batch=batch_idx + 1,
                     size=val_iteration,
                     data=data_time.avg,
                     bt=batch_time.avg,
                     total=bar.elapsed_td,
                     eta=bar.eta_td,
                     loss=losses.avg,
                     train_acc=train_acc.avg)
        bar.next()
    bar.finish()

    return losses.avg, train_acc.avg
