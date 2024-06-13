"""
Main entry point for training and inference

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import pickle as pkl
import sys, os
from typing import Iterable, Optional

from accelerate import Accelerator
from tqdm import tqdm


sys.path.append(os.getcwd())
import hydra
import torch
import wandb
import numpy as np

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras

from configs.structured import ProjectConfig
from trainer import Trainer
import training_utils


class TrainerBehave(Trainer):
    def get_gt_pclouds(self, batch, i):
        ""
        return Pointclouds([batch['pclouds'][i].to('cuda')])

    def get_input_image(self, batch, i):
        return batch['images'][i]

    def get_seq_name(self, batch, i):
        return batch['sequence_name'][i]

    def get_seq_category(self, batch, ind=0):
        return batch['synset_id'][ind]

    def get_metadata(self, batch, i):
        metadata = dict(index=i, sequence_name=batch['sequence_name'][i], sequence_category=batch['synset_id'],
                        frame_timestamp=batch['view_id'][i],
                        camera=self.get_camera(batch, i),
                        image_size_hw=batch['image_size_hw'][i],
                        image_path=batch['image_path'][i],
                        mask_path=batch['image_path'][i],
                        # normalization parameters
                        center=batch['gt_trans'][i],
                        radius=batch['radius'][i],
                        )
        if 'closest_hum' in batch:
            print("Saving closest points")
            # for debug
            metadata.update(closest_hum=batch['closest_hum'][i])
            metadata.update(closest_obj=batch['closest_obj'][i])
            metadata.update(pred_hum=batch['pred_hum'][i])
            metadata.update(pred_obj=batch['pred_obj'][i])
        return metadata

    def get_camera(self, batch, i):
        if self.cfg.dataset.type == 'behave-objonly-segm':
            print("Saving camera using predicted object")
            t = batch['T_obj_scaled'][i][None]
        elif self.cfg.dataset.type == 'behave-humonly-segm':
            print("Saving camera using predicted human")
            t = batch['T_hum_scaled'][i][None]
        else:
            t = batch['T'][i][None]
        cam = PerspectiveCameras(R=batch['R'][i][None],
                                 T=t,
                                 K=batch['K'][i][None],
                                        in_ndc=True,
                                 device='cuda')
        # print("get camera:", cam.K, cam.T)
        return cam

    def get_batch_size(self, batch):
        return len(batch['images'])


class TrainerBinarySegm(TrainerBehave):
    """
    to train a model predict segementation
    """
    def compute_loss(self, batch, model):
        """

        :param batch:
        :param model:
        :return:
        """
        loss, loss_sep = model(batch, mode='train')
        # logging separate losses
        self.loss_sep = loss_sep

        return loss

    def logging_addition(self, log_dict:dict):
        log_dict['train_loss_noise'] = self.loss_sep[0]
        log_dict['train_loss_mse'] = self.loss_sep[1]
        return log_dict

    def add_log_item(self, metric_logger):
        """
        add training losses
        :param metric_logger:
        :return:
        """
        metric_logger.add_meter('train_loss_noise', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('train_loss_mse', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        metric_logger.add_meter('val_loss_noise', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('val_loss_mse', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        return metric_logger


class TrainerCrossAttnHO(TrainerBinarySegm):
    def logging_addition(self, log_dict:dict):
        log_dict['train_loss_hum'] = self.loss_sep[0]
        log_dict['train_loss_obj'] = self.loss_sep[1]
        return log_dict

    def add_log_item(self, metric_logger):
        """
        add training losses
        :param metric_logger:
        :return:
        """
        metric_logger.add_meter('train_loss_hum', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('train_loss_obj', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        metric_logger.add_meter('val_loss_hum', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('val_loss_obj', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

        return metric_logger

    def get_input_image(self, batch, i):
        return batch['images_fullcrop'][i] # return the crop using H+O mask

    def get_gt_pclouds(self, batch, i):
        "rerun center and scale, and combine human + object points to original coordinate "
        pc_h = batch['pclouds'][i] * 2 * batch['radius_hum'][i] + batch['cent_hum'][i]
        pc_o = batch['pclouds_obj'][i] * 2 * batch['radius_obj'][i] + batch['cent_obj'][i]
        return Pointclouds([torch.cat([pc_h, pc_o], 0).to('cuda')])


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    if cfg.model.model_name == 'diff-ho-attn':
        # Stage 2 trainner: train model with separate human and object branch
        trainer = TrainerCrossAttnHO(cfg)
    else:
        if cfg.model.predict_binary:
            trainer = TrainerBinarySegm(cfg)  # use separate model, or predict binary directly
        else:
            trainer = TrainerBehave(cfg)

    if cfg.run.job == 'sample':
        trainer.run_sample(cfg)
    else:
        trainer.train(cfg)


if __name__ == '__main__':
    main()



