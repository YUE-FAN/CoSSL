# CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learningearning

by Yue Fan, Dengxin Dai, Anna Kukleva, Bernt Schiele from MPI-Informatiks.

Paper: https://arxiv.org/abs/2112.04564


## Introduction

Standard semi-supervised learning (SSL) using class-balanced datasets has shown great progress to leverage unlabeled data effectively. However, the more realistic setting of class-imbalanced data - called imbalanced SSL - is largely underexplored and standard SSL tends to underperform. In this paper, we propose a novel co-learning framework (CoSSL), which decouples representation and classifier learning while coupling them closely. To handle the data imbalance, we devise Tail-class Feature Enhancement (TFE) for classifier learning. Furthermore, the current evaluation protocol for imbalanced SSL focuses only on balanced test sets, which has limited practicality in real-world scenarios. Therefore, we further conduct a comprehensive evaluation under various shifted test distributions. In experiments, we show that our approach outperforms other methods over a large range of shifted distributions, achieving state-of-the-art performance on benchmark datasets ranging from CIFAR-10, CIFAR-100, ImageNet, to Food-101.  

<p align="center">
    <img src="images/pipeline.png" width="800"/>
</p>

> Figure: Our co-learning framework CoSSL decouples the training of representation and classifier while coupling them in a non-gradient manner. CoSSL consists of three modules: a semi-supervised representation learning module, a balanced classifier learning module, and a carefully designed pseudo-label generation module. The representation module provides a momentum encoder for feature extraction in the other two modules, and the classifier module produces a balanced classifier using our novel Tail-class Feature Enhancement (TFE). Then, pseudo-label module generates pseudo-labels for the representation module using the momentum encoder and the balanced classifier. The interplay between these modules enhances each other, leading to both a more powerful representation and a more balanced classifier. Additionally, our framework is flexible as it can accommodate any standard SSL methods and classifier learning methods.


## Dependencies

* `python 3.7`
* `pytorch == 1.0.0`
* `torchvision`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`
* `torchvision`
* `progressbar`

## Scripts
Please check out `run*.sh` for the scripts to run the baseline algorithms and ours (CoSSL).

### Training procedure of CoSSL 
Train a network with baseline algorithm, e.g., FixMatch on CIFAR10 at imbalance ratio 150.
```
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 2
```
Applying CoSSL on the baseline algorithm
```
# stage 1
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 2
# stage 2
python train_cifar_fix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint_401.pth.tar --out ./results/cifar10/fixmatch/cossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0
```
## Performance

| Test Acc. (%)      | CIFAR-10-LT γ=50 | CIFAR-10-LT γ=100 | CIFAR-10-LT γ=150 |
|--------------------|------------------|-------------------|-------------------|
| `MixMatch+CoSSL`   | `80.3 ± 0.31`    | `76.4 ± 1.14`     | `73.5 ± 1.25`     |
| `ReMixMatch+CoSSL` | `87.7 ± 0.21`    | `84.1 ± 0.56`     | `81.3 ± 0.83`     |
| `FixMatch+CoSSL`   | `86.8 ± 0.30`    | `83.2 ± 0.49`     | `80.3 ± 0.55`     |

| Test Acc. (%)      | CIFAR-100-LT γ=20 | CIFAR-100-LT γ=50 | CIFAR-100-LT γ=100 |
|--------------------|-------------------|-------------------|--------------------|
| `ReMixMatch+CoSSL` | `55.8 ± 0.62`     | `48.9 ± 0.61`     | `44.1 ± 0.59`      |
| `FixMatch+CoSSL`   | `53.9 ± 0.78`     | `47.6 ± 0.57`     | `43.0 ± 0.61`      |

| Test Acc. (%)      | Food-101-LT γ=50 | Food-101-LT γ=100 |
|--------------------|------------------|-------------------|
| `FixMatch+CoSSL`   | `49.0`           | `40.4`            |

* The performance for the PyTorch version is under checking.

## Citation

Please cite our paper if it is helpful to your work:

```bibtex
@inproceedings{fan2021cossl,
title={CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learning},
author={Fan, Yue and Dai, Dengxin and Kukleva, Anna and Schiele, Bernt},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2022}
}
```

## Acknowledgements

Our implementations use the source code from the following repositories and users:

* [DARP: Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning](https://github.com/bbuing9/DARP)

* [@kekmodel](https://github.com/kekmodel/FixMatch-pytorch)
