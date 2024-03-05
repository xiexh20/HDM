"""
model to deal with shapenet inputs and other datasets such as Behave and ProciGen
the model takes a different data dictionary in forward function
"""
import inspect
from typing import Optional
import numpy as np

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
from tqdm import tqdm
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.datasets.r2n2.utils import BlenderCamera


from .model import ConditionalPointCloudDiffusionModel
from .model_utils import get_num_points


class ConditionalPCDiffusionShapenet(ConditionalPointCloudDiffusionModel):
    def forward(self, batch, mode: str = 'train', **kwargs):
        """
        take a batch of data from ShapeNet
        """
        images = torch.stack(batch['images'], 0).to('cuda')
        masks = torch.stack(batch['masks'], 0).to('cuda')
        pc = Pointclouds([x.to('cuda') for x in batch['pclouds']])
        camera = BlenderCamera(
            torch.stack(batch['R']),
            torch.stack(batch['T']),
            torch.stack(batch['K']), device='cuda'
        )

        if mode == 'train':
            return self.forward_train(
                pc=pc,
                camera=camera,
                image_rgb=images,
                mask=masks,

                **kwargs)
        elif mode == 'sample':
            num_points = kwargs.pop('num_points', get_num_points(pc))
            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                **kwargs)
        else:
            raise NotImplementedError()


class ConditionalPCDiffusionBehave(ConditionalPointCloudDiffusionModel):
    "diffusion model for Behave dataset"
    def forward(self, batch, mode: str = 'train', **kwargs):
        images = torch.stack(batch['images'], 0).to('cuda')
        masks = torch.stack(batch['masks'], 0).to('cuda')
        pc = self.get_input_pc(batch)
        camera = PerspectiveCameras(
            R=torch.stack(batch['R']),
            T=torch.stack(batch['T']),
            K=torch.stack(batch['K']),
            device='cuda',
            in_ndc=True
        )
        grid_df = torch.stack(batch['grid_df'], 0).to('cuda') if 'grid_df' in batch else None
        num_points = kwargs.pop('num_points', get_num_points(pc))
        if mode == 'train':
            return self.forward_train(
                pc=pc,
                camera=camera,
                image_rgb=images,
                mask=masks,
                grid_df=grid_df,
                **kwargs)
        elif mode == 'sample':
            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                **kwargs)
        else:
            raise NotImplementedError()

    def get_input_pc(self, batch):
        pc = Pointclouds([x.to('cuda') for x in batch['pclouds']])
        return pc


class ConditionalPCDiffusionSeparateSegm(ConditionalPCDiffusionBehave):
    "a separate model to predict binary labels, the final segmentation model"
    def __init__(self,
                 beta_start: float,
                 beta_end: float,
                 beta_schedule: str,
                 point_cloud_model: str,
                 point_cloud_model_embed_dim: int,
                 **kwargs,  # projection arguments
                 ):
        super(ConditionalPCDiffusionSeparateSegm, self).__init__(beta_start, beta_end, beta_schedule,
                                                                 point_cloud_model,
                                                                 point_cloud_model_embed_dim, **kwargs)
        # add a separate model to predict binary label
        from .point_cloud_transformer_model import PointCloudTransformerModel, PointCloudModel

        self.binary_model = PointCloudTransformerModel(
            num_layers=1, # XH: use the default color model number of layers
            model_type=point_cloud_model, # pvcnn
            embed_dim=point_cloud_model_embed_dim, # save as pc shape model
            in_channels=self.in_channels,
            out_channels=1,
        )
        self.binary_training_noise_std = kwargs.get("binary_training_noise_std", 0.1)

        # re-initialize point cloud model
        assert self.predict_binary
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels - 1,  # not predicting binary from this anymore
            voxel_resolution_multiplier=kwargs.get('voxel_resolution_multiplier', 1)
        )

    def forward_train(
        self,
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        # first run shape forward, then binary label forward
        assert not return_intermediate_steps
        assert self.predict_binary
        loss_shape = super(ConditionalPCDiffusionSeparateSegm, self).forward_train(pc,
                                                                                   camera,
                                                                                   image_rgb,
                                                                                   mask,
                                                                                   return_intermediate_steps,
                                                                                   **kwargs)

        # binary label forward
        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
        x_points, x_colors = x_0[:, :, :3], x_0[:, :, 3:]

        # Add noise to points.
        x_input = x_points + torch.randn_like(x_points) * self.binary_training_noise_std # std=0.1
        x_input = self.get_input_with_conditioning(x_input, camera=camera,
                                                   image_rgb=image_rgb, mask=mask, t=None)

        # Forward
        pred_segm = self.binary_model(x_input)

        # use compressed bits
        df_grid = kwargs.get('grid_df', None).unsqueeze(1)  # (B, 1, resz, resy, resx)
        points = x_points.clone().detach() / self.scale_factor * 2  # , normalize to [-1, 1]
        points[:, :, 0], points[:, :, 2] = points[:, :, 2].clone(), points[:, :,0].clone()  # swap, make sure clone is used!
        points = points.unsqueeze(1).unsqueeze(1)  # (B,1, 1, N, 3)
        with torch.no_grad():
            df_interp = F.grid_sample(df_grid, points, padding_mode='border', align_corners=True).squeeze(1).squeeze(1)  # (B, 1, 1, 1, N)
        binary_label = df_interp[:, 0] > 0.5  # (B, 1, N)

        binary_pred = torch.sigmoid(pred_segm.squeeze(-1))  # add a sigmoid layer
        loss_binary = F.mse_loss(binary_pred, binary_label.float().squeeze(1).squeeze(1)) * self.lw_binary
        loss = loss_shape + loss_binary

        return loss, torch.tensor([loss_shape, loss_binary])

    def reverse_step(self, extra_step_kwargs, scheduler, t, x_t, x_t_input, **kwargs):
        "return (B, N, 4), the 4-th channel is binary label"
        B = x_t.shape[0]
        # Forward
        noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))
        if self.consistent_center:
            assert self.dm_pred_type != 'sample', 'incompatible dm predition type!'
            # suggested by the CCD-3DR paper
            noise_pred = noise_pred - torch.mean(noise_pred, dim=1, keepdim=True)
        # Step: make sure only update the shape (first 3 channels)
        x_t = scheduler.step(noise_pred, t, x_t[:, :, :3], **extra_step_kwargs).prev_sample
        if self.consistent_center:
            x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)

        # also add binary prediction
        if kwargs.get('inference_binary', False):
            pred_segm = self.binary_model(x_t_input)
        else:
            pred_segm = torch.zeros_like(x_t[:, :, 0:1])

        x_t = torch.cat([x_t, torch.sigmoid(pred_segm)], -1)

        return x_t

    def get_coord_feature(self, x_t):
        x_t_input = [x_t[:, :, :3]]
        return x_t_input

    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        """
        take binary label into account
        :param self:
        :param x: (B, N, 4), the 4th channel is the binary segmentation, 1-human, 0-object
        :param denormalize: denormalize the per-point colors, from pc2
        :param unscale: undo point scaling, from pc2
        :return: pc with point colors if predict binary label or per-point color
        """
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        if self.predict_color:
            colors = self.denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
            return Pointclouds(points=points, features=colors)
        else:
            if self.predict_binary:
                assert x.shape[2] == 4
                # add color to predicted binary labels
                is_hum = x[:, :, 3] > 0.5
                features = []
                for mask in is_hum:
                    color = torch.zeros_like(x[0, :, :3]) + torch.tensor([0.5, 1.0, 0]).to(x.device)
                    color[mask, :] = torch.tensor([0.05, 1.0, 1.0]).to(x.device) # human is light blue, object light green
                    features.append(color)
            else:
                assert x.shape[2] == 3
                features = None
            return Pointclouds(points=points, features=features)


