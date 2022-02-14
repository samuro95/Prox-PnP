# Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization

[[Paper](https://arxiv.org/abs/2201.13256)]

[Samuel Hurault](https://www.math.u-bordeaux.fr/~shurault/), [Arthur Leclaire](https://www.math.u-bordeaux.fr/~aleclaire/), [Nicolas Papadakis](https://www.math.u-bordeaux.fr/~npapadak/). \
[Institut de Mathématiques de Bordeaux](https://www.math.u-bordeaux.fr/imb/spip.php), France.


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
python main_train.py --name GS_DRUNet
```
Checkpoints, tensorboard events and hyperparameters will be saved in the ```GS_denoising/logs/GS_DRUNet``` subfolder. 

- Save the trained model in the ckpts directory :  
```
cp logs/GS_DRUNet/version_0/checkpoints/* ckpts/GS_DRUNet.ckpt
```

- Finetune previous training constraining the spectral norm (15 epochs) : 
```
python main_train.py --name Prox_DRUNet  --resume_from_checkpoint --pretrained_checkpoint ckpts/GS_DRUNet.ckpt --jacobian_loss_weight 1e-3 
```
- Save the trained model in the ckpts directory :  
```
cp logs/Prox_DRUNet/version_0/checkpoints/* ckpts/Prox_DRUNet.ckpt
```


## Acknowledgments

This repo contains parts of code taken from : 
- Deep Plug-and-Play Image Restoration (DPIR) : https://github.com/cszn/DPIR 
- Gradient Step Denoiser for convergent Plug-and-Play (GS-PnP) : https://github.com/samuro95/GSPnP

## Citation 
```
@article{hurault2022proximal,
  title={Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization},
  author={Hurault, Samuel and Leclaire, Arthur and Papadakis, Nicolas},
  journal={arXiv preprint arXiv:2201.13256},
  year={2022}
}

```
