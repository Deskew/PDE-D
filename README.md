# Towards Efficient Motion Video SCI via Progressive Degradation Estimation and Denoising
## Abstract
Video snapshot compressive imaging (SCI)

## Testing Result on Six Simulation Dataset
|Dataset|Kobe |Traffic|Runner| Drop  |Aerial|Vehicle|Average|
|:----:|:----:|:----: |:-----:|:----:|:----:|:----:|:----:|
|PSNR  |:----:|:-----:|:-----:|:----:|:----:|:----:|:----:| 
|SSIM  |:----:|:-----:|:-----:|:----:|:----:|:----:|:----:|

## Multi Platform Running Time Analysis 
|GTX 1080ti |RTX 3080 |RTX 3090 | RTX8000 | RTX A40|
|:---------:|:------: |:-------:|:-------:|:------:|
|:---------:|:-------:|:-------:|:-------:|:------:|

## Training PDE-D 
Support multi GPUs and single GPU training efficiently, first configure the training dataset based on [model training dataset](../../docs/add_datasets.md).

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/MIXST_SCI/mixst_sci.py --distributed=True
```
* CUDA_VISIBLE_DEVICE: specify number of GPUs
* --nproc_per_node: number of used GPUs
* --master_port: main node port number, usually for communication

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/MIXST_SCI/mixst_sci.py
```

## Testing PDE-D on Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in simulation dataset by executing the statement below.

```
python tools/test.py configs/MIXST_SCI/mixst_sci.py --weights=checkpoints/pde-d/PDE-D-5st_gray.pth
```
* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [dropbox](https://www.dropbox.com/sh/96nf7jzabhqj4mh/AAB09QXrNGi_kujDDnWn6G32a?dl=0).


## Testing PDE-D on Real Dataset 
Launch PDE-D on real dataset by executing the statement below.

```
python tools/real_data/test.py configs/MIXST_SCI/mixst_sci_real_cr10.py --weights=checkpoints/pde-d/PDE-D-5st_real_cr10.pth

```
Notice:

* --weights: path of weighted parameters
  Notice: path of weighted parameters can be specified by --weight, also can be set by modifying checkpoints value in the configuration file, related weight can be download via [Google Drive](https://drive.google.com/drive/folders/1PWsXRfzLKuH0BeqjsLshVnx5JH_BvaHx?usp=sharing).
* Experiment Results can be download via [Google Drive](https://drive.google.com/drive/folders/1AS3tUeAsTxlAguSVwX2Xr2EQ8UO0nXo9?usp=sharing).

## Citation
```


```
## Acknowledgement
The codes are based on [CACTI](https://github.com/ucaswangls/cacti), 
we also refer to codes in [DADUN](https://github.com/Deskew/DADUN.git), 
and [RDLUF_MixS2](https://github.com/ShawnDong98/RDLUF_MixS2). Thanks for their awesome works.
