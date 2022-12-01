# Constrained CycleGAN for Effective Generation of Ultrasound Sector Images of Uniform Spatial Resolution

Xiaofei Sun

The University of Hong Kong


## Introduction

Cycle-consistent adversarial networks (CycleGAN) has been widely used for image conversions. This is an implementation of Constrained CycleGAN on ultrasound images generation for improving the spatial resolutions.

<p align="center">
    <img src = "./Figures/1.png" width="88%">
</p>

## Notice:

       This repo contains no training dataset due to privacy concern. Codes are for your reference.
       The CCyclegan model used in the article is in the folder 'CCycleGAN Eexample Model'.  

## Dependencies

* Python 3.7
* TensorFlow 2.0.0 above
* opencv-python 3.4.14
* matplotlib 3.3.4 

## Files

```
.
├── CCYCLEGAN
  ├── code
    ├── ccyclegan_model.py
    ├── ccyclegan_module.py
    ├── main.py
  ├── data
├── CCycleGAN Eexample Model
├── Figures
  ├── 1.png
├── LICENSE.md
├── inout_util_mat.py
├── README.md
```

## Usage

Because the model was implemented using TensorFlow 2.0, there could be some warnings due to function deprecations when running the programs.

### Train Model
In main.py , set args.phase = 'train'.
To have a good generation capability, the training would take at least 100 epochs. 

```bash
$ python main.py --help
usage: main.py [-h] [--data_path MAT_PATH] [--LRUS_path LRUS_PATH]
               [--HRUS_path HRUS_PATH] [--LRUS_val_path LRUS_VAL_PATH]
               [--HRUS_val_path HRUS_VAL_PATH]
               [--LRUS_test_path LRUS_TEST_PATH]
               [--HRUS_test_path HRUS_TEST_PATH] [--data_info DATA_INFO]
               [--checkpoint_dir CHECKPOINT_DIR]
               [--test_npy_save_dir TEST_NPY_SAVE_DIR]
               [--patch_size PATCH_SIZE] [--whole_size WHOLE_SIZE]
               [--img_channel IMG_CHANNEL] [--img_vmax IMG_VMAX]
               [--img_vmin IMG_VMIN] [--model MODEL] [--phase PHASE]
               [--end_epoch END_EPOCH] [--decay_epoch DECAY_EPOCH] [--lr LR]
               [--batch_size BATCH_SIZE] [--L1_lambda1 L1_LAMBDA1]
               [--L1_lambda2 L1_LAMBDA2] [--L1_lambda3 L1_LAMBDA3]
               [--beta1 BETA1] [--beta2 BETA2] [--ngf NGF] [--nglf NGLF]
               [--ndf NDF] [--save_freq SAVE_FREQ] [--print_freq PRINT_FREQ]
               [--continue_train CONTINUE_TRAIN] [--gpu_no GPU_NO]
               [--unpair UNPAIR]
```

### Test Model
In main.py , set args.phase = 'test'











