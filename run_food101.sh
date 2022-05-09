
# Food-101@r=50
CUDA_VISIBLE_DEVICES=0,1 python train_food_fix.py --ratio 2 --num_max 250 --imb_ratio_l 50 --imb_ratio_u 50 --out ./results/food/fixmatch/baseline/resnet50_N250_r50 --epochs 1000 --decay_epochs 9000 --batch-size 256 --tau 0.90
CUDA_VISIBLE_DEVICES=0,1 python train_food_fix_cossl.py --ratio 2 --num_max 250 --imb_ratio_l 50 --imb_ratio_u 50 --resume ./results/food/fixmatch/baseline/resnet50_N250_r50/checkpoint_800.pth.tar --out ./results/food/fixmatch/cossl/resnet50_N250_r50 --epochs 200 --batch-size 256 --tau 0.90

# Food-101@r=100
CUDA_VISIBLE_DEVICES=0,1 python train_food_fix.py --ratio 2 --num_max 250 --imb_ratio_l 100 --imb_ratio_u 100 --out ./results/food/fixmatch/baseline/resnet50_N250_r100 --epochs 1000 --decay_epochs 9000 --batch-size 256 --tau 0.90
CUDA_VISIBLE_DEVICES=0,1 python train_food_fix_cossl.py --ratio 2 --num_max 250 --imb_ratio_l 100 --imb_ratio_u 100 --resume ./results/food/fixmatch/baseline/resnet50_N250_r100/checkpoint_800.pth.tar --out ./results/food/fixmatch/cossl/resnet50_N250_r100 --epochs 200 --batch-size 256 --tau 0.90
