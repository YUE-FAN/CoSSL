# Prepare Small-ImageNet-127
This directory contains instructions for constructing Small-ImageNet-127 we used in the paper.


## Prerequisites

Follow the instructions from [here](https://patrykchrabaszcz.github.io/Imagenet32/)
 and download Small-ImageNet 32 & 64.

Both archives have the following structure:

    .
    ├── train_data_batch_1
    ├── train_data_batch_2
    ├── train_data_batch_3
    ├── train_data_batch_4
    ├── train_data_batch_5
    ├── train_data_batch_6
    ├── train_data_batch_7
    ├── train_data_batch_8
    ├── train_data_batch_9
    ├── train_data_batch_10
    └── val_data




## Construct Small-ImageNet-127

1. Open ``construct_small_imagenet_127.py`` and change the following two lines into your downloaded Small-ImageNet location and your preferred save path of the new datasets.

   ```
   small_imagenet_path = 'PATH TO SMALL Imagenet32/64'
   small_imagenet_127_save_path = 'PATH TO SAVE YOUR SMALL Imagenet 127 32/64'
   ```

3. Run ``construct_small_imagenet_127.py``.
