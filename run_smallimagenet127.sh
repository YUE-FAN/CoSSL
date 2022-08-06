
# Small-ImageNet@32x32
python small_imagenet127_fix.py --labeled_percent 0.1 --img_size 32 --epoch 500 --val-iteration 500 --out ./results/small_imagenet127_32x32/fixmatch/baseline/resnet50_labeled_percent01 --gpu 0
python small_imagenet127_fix_cossl.py --labeled_percent 0.1 --img_size 32 --epoch 100 --val-iteration 500 --resume ./results/small_imagenet127_32x32/fixmatch/baseline/resnet50_labeled_percent01/checkpoint_401.pth.tar --out ./results/small_imagenet127_32x32/fixmatch/cossl/resnet50_labeled_percent01 --max_lam 0.6 --gpu 0

# Small-ImageNet@64x64
python small_imagenet127_fix.py --labeled_percent 0.1 --img_size 64 --epoch 500 --val-iteration 500 --out ./results/small_imagenet127_64x64/fixmatch/baseline/resnet50_labeled_percent01 --gpu 0
python small_imagenet127_fix_cossl.py --labeled_percent 0.1 --img_size 64 --epoch 100 --val-iteration 500 --resume ./results/small_imagenet127_64x64/fixmatch/baseline/resnet50_labeled_percent01/checkpoint_401.pth.tar --out ./results/small_imagenet127_64x64/fixmatch/cossl/resnet50_labeled_percent01 --max_lam 0.6 --gpu 0
