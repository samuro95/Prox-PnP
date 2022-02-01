# Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization

[[Paper](https://arxiv.org/abs/2201.13256)]

[Samuel Hurault](https://www.math.u-bordeaux.fr/~shurault/), [Arthur Leclaire](https://www.math.u-bordeaux.fr/~aleclaire/), [Nicolas Papadakis](https://www.math.u-bordeaux.fr/~npapadak/). \
[Institut de Mathématiques de Bordeaux](https://www.math.u-bordeaux.fr/imb/spip.php), France.

<img src="images/results.png" width="800px"/> 

## Prerequisites


The code was computed with Python 3.8.10, PyTorch Lightning 1.2.6, PyTorch 1.7.1

```
pip install -r requirements.txt
```

## Prox-Denoiser (Prox-DRUNet)

The code relative to the Proximal (Gradient Step) Denoiser can be found in the ```GS_denoising``` directory.

### Training 

- Download training dataset from https://drive.google.com/file/d/1WVTgEBZgYyHNa2iVLUYwcrGWZ4LcN4--/view?usp=sharing and unzip ```DRUNET``` in the ```datasets``` folder

- Realize a first baseline training of the Gradient Step denoiser without constraining the spectral norm (1200 epochs) :
```
cd GS_denoising
python main_train.py --name GS_denoiser --log_folder logs
```
Checkpoints, tensorboard events and hyperparameters will be saved in the ```GS_denoising/logs/experiment_name``` subfolder. 

- Save the trained model in the ckpts directory :  
```
cp logs/GS_denoiser/version_0/checkpoints/* ckpts/GS_denoiser.ckpt
```

- Finetune previous training constraining the spectral norm (15 epochs) : 
```
python main_train.py --name Prox_denoiser --log_folder logs --resume_from_checkpoint --pretrained_checkpoint ckpts/GS_denoiser.ckpt --jacobian_loss_weight 1e-3 
```


### Testing 

- Download pretrained checkpoint from https://drive.google.com/file/d/1aafXsJG50FHdNIBYfQZ2jRKTfY0ig6Zi/view?usp=sharing and save it as ```GS_denoising/ckpts/Prox-DRUNet.ckpt```
- For denoising the whole CBSD68 dataset at input Gaussian noise level 25 :
```
cd PnP_restoration
python denoise.py --dataset_name CBSD68 --noise_level_img 25
```
Add the argument ```--extract_images``` the save the output images.




## Acknowledgments

This repo contains parts of code taken from : 
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 
- Gradient Step Denoiser for convergent Plug-and-Play (GS-PnP) : https://github.com/samuro95/GSPnP

## Citation 
```
@inproceedings{
hurault2022gradient,
title={Gradient Step Denoiser for convergent Plug-and-Play},
author={Samuel Hurault and Arthur Leclaire and Nicolas Papadakis},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=fPhKeld3Okz}
}

```
