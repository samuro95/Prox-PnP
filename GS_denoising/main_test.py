import pytorch_lightning as pl
from lightning_denoiser import GradMatch
from data_module import DataModule
from pytorch_lightning import loggers as pl_loggers
import os
from argparse import ArgumentParser
import torch

def main() :

    # PROGRAM args
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--save_images', dest='save_images', action='store_true')
    parser.set_defaults(save_images=False)

    # MODEL args
    parser = GradMatch.add_model_specific_args(parser)
    # DATA args
    parser = DataModule.add_data_specific_args(parser)
    # OPTIM args
    parser = GradMatch.add_optim_specific_args(parser)

    hparams = parser.parse_args()

    log_path = 'test_logs/' + hparams.name

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    tb_logger = pl_loggers.TensorBoardLogger(log_path)

    model = GradMatch(hparams)
    dm = DataModule(hparams)

    checkpoint = torch.load(hparams.pretrained_checkpoint, map_location=model.device)
    model.load_state_dict(checkpoint['state_dict'],strict=False)


    n_gpus = 1

    trainer = pl.Trainer.from_argparse_args(hparams, logger=tb_logger, gpus=n_gpus, accelerator='ddp')

    result = trainer.test(model,datamodule=dm)

    return result


if __name__ == '__main__':
    main()




