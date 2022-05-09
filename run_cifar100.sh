

# ReMixMatch@r=20
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/seed/reproduce/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 2
python train_cifar_remix_cossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r20_seed1/checkpoint_401.pth.tar --out ./results/cifar100/remixmatch/seed/cossl/wrn28_N150_r20_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 2

# ReMixMatch@r=50
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/seed/reproduce/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 2
python train_cifar_remix_cossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r50_seed1/checkpoint_401.pth.tar --out ./results/cifar100/remixmatch/seed/cossl/wrn28_N150_r50_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 2

# ReMixMatch@r=100
python train_cifar_remix.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar100/remixmatch/seed/reproduce/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 2
python train_cifar_remix_cossl.py --dataset cifar100 --align --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar100/remixmatch/baseline/wrn28_N150_r100_seed1/checkpoint_401.pth.tar --out ./results/cifar100/remixmatch/seed/cossl/wrn28_N150_r100_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 2


# FixMatch@r=20
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/seed/reproduce/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 2
python train_cifar_fix_cossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r20_seed1/checkpoint_401.pth.tar --out ./results/cifar100/fixmatch/seed/cossl/wrn28_N150_r20_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 2

# FixMatch@r=50
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/seed/reproduce/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 2
python train_cifar_fix_cossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r50_seed1/checkpoint_401.pth.tar --out ./results/cifar100/fixmatch/seed/cossl/wrn28_N150_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 2

# FixMatch@r=100
python train_cifar_fix.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar100/fixmatch/seed/reproduce/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 2
python train_cifar_fix_cossl.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar100/fixmatch/baseline/wrn28_N150_r100_seed1/checkpoint_401.pth.tar --out ./results/cifar100/fixmatch/seed/cossl/wrn28_N150_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 2
