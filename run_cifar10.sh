
# MixMatch@r=50
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r50_seed1/ --manualSeed 1 --gpu 0
python train_cifar_mix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint_401.pth.tar --out ./results/cifar10/mixmatch/cossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=100
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r100_seed1/ --manualSeed 1 --gpu 0
python train_cifar_mix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint_401.pth.tar --out ./results/cifar10/mixmatch/cossl/wrn28_N1500_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# MixMatch@r=150
python train_cifar_mix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/mixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0
python train_cifar_mix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/mixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint_401.pth.tar --out ./results/cifar10/mixmatch/cossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# ReMixMatch@r=50
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 1
python train_cifar_remix_cossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint_401.pth.tar --out ./results/cifar10/remixmatch/cossl/wrn28_N1500_r50_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=100
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 1
python train_cifar_remix_cossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint_401.pth.tar --out ./results/cifar10/remixmatch/cossl/wrn28_N1500_r100_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# ReMixMatch@r=150
python train_cifar_remix.py --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/remixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 1
python train_cifar_remix_cossl.py --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/remixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint_401.pth.tar --out ./results/cifar10/remixmatch/cossl/wrn28_N1500_r150_lam06_seed1 --manualSeed 1 --max_lam 0.6 --gpu 0

# FixMatch@r=50
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1 --manualSeed 1 --gpu 2
python train_cifar_fix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r50_seed1/checkpoint_401.pth.tar --out ./results/cifar10/fixmatch/cossl/wrn28_N1500_r50_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=100
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 2
python train_cifar_fix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r100_seed1/checkpoint_401.pth.tar --out ./results/cifar10/fixmatch/cossl/wrn28_N1500_r100_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0

# FixMatch@r=150
python train_cifar_fix.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 2
python train_cifar_fix_cossl.py --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 100 --val-iteration 500 --resume ./results/cifar10/fixmatch/baseline/wrn28_N1500_r150_seed1/checkpoint_401.pth.tar --out ./results/cifar10/fixmatch/cossl/wrn28_N1500_r150_lam06_seed1 --max_lam 0.6 --manualSeed 1 --gpu 0
