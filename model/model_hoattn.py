"""
model that use cross attention to predict human + object
"""

import inspect
import random
from typing import Optional
from torch import Tensor
import torch
import numpy as np

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import CamerasBase
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from .model_diff_data import ConditionalPCDiffusionBehave
from .pvcnn.pvcnn_ho import PVCNN2HumObj
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras
from .model_utils import get_num_points
from tqdm import tqdm


class CrossAttenHODiffusionModel(ConditionalPCDiffusionBehave):
    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        """use cross attention model"""
        if point_cloud_model == 'pvcnn':
            self.point_cloud_model = PVCNN2HumObj(embed_dim=point_cloud_model_embed_dim,
                                        num_classes=self.out_channels,
                                        extra_feature_channels=(self.in_channels - 3),
                                        voxel_resolution_multiplier=kwargs.get('voxel_resolution_multiplier', 1),
                                        attn_type=kwargs.get('attn_type', 'simple-cross'),
                                        attn_weight=kwargs.get("attn_weight", 1.0)
                                 )
        else:
            raise ValueError(f"Unknown point cloud model {point_cloud_model}!")
        self.point_visible_test = kwargs.get("point_visible_test", 'single') # when doing point visibility test, use only human points or human + object?
        assert self.point_visible_test in ['single', 'combine'], f'invalide point visible test option {self.point_visible_test}'
        # print(f"Point visibility test is based on {self.point_visible_test} point clouds!")

    def forward_train(
        self,
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        "additional input (RGB, mask, camera, and pc) for object is read from kwargs"
        # assert not self.consistent_center
        assert not self.self_conditioning

        # Normalize colors and convert to tensor
        x0_h = self.point_cloud_to_tensor(pc, normalize=True, scale=True)  # this will not pack the point colors
        x0_o = self.point_cloud_to_tensor(kwargs.get('pc_obj'), normalize=True, scale=True)
        B, N, D = x0_h.shape

        # Sample random noise
        noise = torch.randn_like(x0_h)
        if self.consistent_center:
            # modification suggested by https://arxiv.org/pdf/2308.07837.pdf
            noise = noise - torch.mean(noise, dim=1, keepdim=True)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,),
                                 device=self.device, dtype=torch.long)
        # timestep = torch.randint(0, 1, (B,),
        #                          device=self.device, dtype=torch.long)

        # Add noise to points
        xt_h = self.scheduler.add_noise(x0_h, noise, timestep)
        xt_o = self.scheduler.add_noise(x0_o, noise, timestep)
        norm_parms = self.pack_norm_params(kwargs) # (2, B, 4)

        # get input conditioning
        x_t_input_h, x_t_input_o = self.get_image_conditioning(camera, image_rgb, kwargs, mask, norm_parms, timestep,
                                                               xt_h, xt_o)

        # Diffusion prediction
        noise_pred_h, noise_pred_o = self.point_cloud_model(x_t_input_h, x_t_input_o, timestep, norm_parms)

        # Check
        if not noise_pred_h.shape == noise.shape:
            raise ValueError(f'{noise_pred_h.shape=} and {noise.shape=}')
        if not noise_pred_o.shape == noise.shape:
            raise ValueError(f'{noise_pred_o.shape=} and {noise.shape=}')

        # Loss
        loss_h = F.mse_loss(noise_pred_h, noise)
        loss_o = F.mse_loss(noise_pred_o, noise)

        loss = loss_h + loss_o

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x0_h, xt_h, noise, noise_pred_h)

        return loss, torch.tensor([loss_h, loss_o])

    def get_image_conditioning(self, camera, image_rgb, kwargs, mask, norm_parms, timestep, xt_h, xt_o):
        """
        compute image features for each point
        :param camera:
        :param image_rgb:
        :param kwargs:
        :param mask:
        :param norm_parms:
        :param timestep:
        :param xt_h:
        :param xt_o:
        :return:
        """
        if self.point_visible_test == 'single':
            # Visibility test is down independently for human and object
            x_t_input_h = self.get_input_with_conditioning(xt_h, camera=camera,
                                                           image_rgb=image_rgb, mask=mask, t=timestep)
            x_t_input_o = self.get_input_with_conditioning(xt_o, camera=kwargs.get('camera_obj'),
                                                           image_rgb=kwargs.get('rgb_obj'),
                                                           mask=kwargs.get('mask_obj'), t=timestep)
        elif self.point_visible_test == 'combine':
            # Combine human + object points to do visibility test and obtain features
            B, N = xt_h.shape[:2]  # (B, N, 3)
            # for human: transform object points first to H+O space, then to human space
            xt_o_in_ho = xt_o * 2 * norm_parms[1, :, 3:].unsqueeze(1) + norm_parms[1, :, :3].unsqueeze(1)
            xt_o_in_hum = (xt_o_in_ho - norm_parms[0, :, :3].unsqueeze(1)) / (2 * norm_parms[0, :, 3:].unsqueeze(1))
            # compute features for all points, take only first half feature for human
            x_t_input_h = self.get_input_with_conditioning(torch.cat([xt_h, xt_o_in_hum], 1), camera=camera,
                                                           image_rgb=image_rgb, mask=mask, t=timestep)[:,:N]
            # for object: transform human points to H+O space, then to object space
            xt_h_in_ho = xt_h * 2 * norm_parms[0, :, 3:].unsqueeze(1) + norm_parms[0, :, :3].unsqueeze(1)
            xt_h_in_obj = (xt_h_in_ho - norm_parms[1, :, :3].unsqueeze(1)) / (2 * norm_parms[1, :, 3:].unsqueeze(1))
            x_t_input_o = self.get_input_with_conditioning(torch.cat([xt_o, xt_h_in_obj], 1),
                                                           camera=kwargs.get('camera_obj'),
                                                           image_rgb=kwargs.get('rgb_obj'),
                                                           mask=kwargs.get('mask_obj'), t=timestep)[:, :N]
        else:
            raise NotImplementedError
        return x_t_input_h, x_t_input_o

    def forward(self, batch, mode: str = 'train', **kwargs):
        """"""
        images = torch.stack(batch['images'], 0).to('cuda')
        masks = torch.stack(batch['masks'], 0).to('cuda')
        pc = self.get_input_pc(batch)
        camera = PerspectiveCameras(
            R=torch.stack(batch['R']),
            T=torch.stack(batch['T_hum']),
            K=torch.stack(batch['K_hum']),
            device='cuda',
            in_ndc=True
        )
        grid_df = torch.stack(batch['grid_df'], 0).to('cuda') if 'grid_df' in batch else None
        num_points = kwargs.pop('num_points', get_num_points(pc))

        rgb_obj = torch.stack(batch['images_obj'], 0).to('cuda')
        masks_obj = torch.stack(batch['masks_obj'], 0).to('cuda')
        pc_obj = Pointclouds([x.to('cuda') for x in batch['pclouds_obj']])
        camera_obj = PerspectiveCameras(
            R=torch.stack(batch['R']),
            T=torch.stack(batch['T_obj']),
            K=torch.stack(batch['K_obj']),
            device='cuda',
            in_ndc=True
        )

        # normalization parameters
        cent_hum = torch.stack(batch['cent_hum'], 0).to('cuda')
        cent_obj = torch.stack(batch['cent_obj'], 0).to('cuda') # B, 3
        radius_hum = torch.stack(batch['radius_hum'], 0).to('cuda') # B, 1
        radius_obj = torch.stack(batch['radius_obj'], 0).to('cuda')

        # print(batch['image_path'])

        if mode == 'train':
            return self.forward_train(
                pc=pc,
                camera=camera,
                image_rgb=images,
                mask=masks,
                grid_df=grid_df,
                rgb_obj=rgb_obj,
                mask_obj=masks_obj,
                pc_obj=pc_obj,
                camera_obj=camera_obj,
                cent_hum=cent_hum,
                cent_obj=cent_obj,
                radius_hum=radius_hum,
                radius_obj=radius_obj,
            )
        elif mode == 'sample':
            # this use GT centers to do projection
            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                rgb_obj=rgb_obj,
                mask_obj=masks_obj,
                pc_obj=pc_obj,
                camera_obj=camera_obj,
                cent_hum=cent_hum,
                cent_obj=cent_obj,
                radius_hum=radius_hum,
                radius_obj=radius_obj,
                **kwargs)
        elif mode == 'interm-gt':
            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                rgb_obj=rgb_obj,
                mask_obj=masks_obj,
                pc_obj=pc_obj,
                camera_obj=camera_obj,
                cent_hum=cent_hum,
                cent_obj=cent_obj,
                radius_hum=radius_hum,
                radius_obj=radius_obj,
                sample_from_interm=True,
                **kwargs)
        elif mode == 'interm-pred':
            # use camera from predicted
            camera = PerspectiveCameras(
                R=torch.stack(batch['R']),
                T=torch.stack(batch['T_hum_scaled']),
                K=torch.stack(batch['K_hum']),
                device='cuda',
                in_ndc=True
            )
            camera_obj = PerspectiveCameras(
                R=torch.stack(batch['R']),
                T=torch.stack(batch['T_obj_scaled']),
                K=torch.stack(batch['K_obj']), # the camera should be human/object specific!!!
                device='cuda',
                in_ndc=True
            )
            # use pc from predicted
            pc = Pointclouds([x.to('cuda') for x in batch['pred_hum']])
            pc_obj = Pointclouds([x.to('cuda') for x in batch['pred_obj']])
            # use center and radius from predicted
            cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
            cent_obj = torch.stack(batch['cent_obj_pred'], 0).to('cuda')  # B, 3
            radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')  # B, 1
            radius_obj = torch.stack(batch['radius_obj_pred'], 0).to('cuda')

            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                rgb_obj=rgb_obj,
                mask_obj=masks_obj,
                pc_obj=pc_obj,
                camera_obj=camera_obj,
                cent_hum=cent_hum,
                cent_obj=cent_obj,
                radius_hum=radius_hum,
                radius_obj=radius_obj,
                sample_from_interm=True,
                **kwargs)
        elif mode == 'interm-pred-ts':
            # use only estimate translation and scale, but sample from gaussian
            # this works, the camera is GT!!!
            pc = Pointclouds([x.to('cuda') for x in batch['pred_hum']])
            pc_obj = Pointclouds([x.to('cuda') for x in batch['pred_obj']])
            # use center and radius from predicted
            cent_hum = torch.stack(batch['cent_hum_pred'], 0).to('cuda')
            cent_obj = torch.stack(batch['cent_obj_pred'], 0).to('cuda')  # B, 3
            radius_hum = torch.stack(batch['radius_hum_pred'], 0).to('cuda')  # B, 1
            radius_obj = torch.stack(batch['radius_obj_pred'], 0).to('cuda')
            # print(cent_hum[0], radius_hum[0], cent_obj[0], radius_obj[0])

            return self.forward_sample(
                num_points=num_points,
                camera=camera,
                image_rgb=images,
                mask=masks,
                gt_pc=pc,
                rgb_obj=rgb_obj,
                mask_obj=masks_obj,
                pc_obj=pc_obj,
                camera_obj=camera_obj,
                cent_hum=cent_hum,
                cent_obj=cent_obj,
                radius_hum=radius_hum,
                radius_obj=radius_obj,
                sample_from_interm=False,
                **kwargs)
        else:
            raise NotImplementedError

    def forward_sample(
        self,
        num_points: int,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = 'ddpm',
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        gt_pc: Pointclouds = None,
            **kwargs
    ):
        "use two models to run diffusion forward, and also use translation and scale to put them back"
        assert not self.self_conditioning
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        B = 1 if image_rgb is None else image_rgb.shape[0]
        D = self.get_x_T_channel()
        device = self.device if image_rgb is None else image_rgb.device

        # sample from full steps or only a few steps
        sample_from_interm = kwargs.get('sample_from_interm', False)
        interm_steps = kwargs.get('noise_step') if sample_from_interm else -1

        xt_h = self.initialize_x_T(device, gt_pc, (B, N, D), interm_steps, scheduler)
        xt_o = self.initialize_x_T(device, kwargs.get('pc_obj', None), (B, N, D), interm_steps, scheduler)

        # the segmentation mask
        segm_mask = torch.zeros(B, 2*N, 1).to(device)
        segm_mask[:, :N] = 1.0

        # Set timesteps
        extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(self.get_reverse_timesteps(scheduler, interm_steps),
                            desc=f'Sampling ({xt_h.shape})', disable=disable_tqdm)

        # print("Camera T:", camera.T[0], camera.R[0])
        # print("Camera_obj T:", kwargs.get('camera_obj').T[0], kwargs.get('camera_obj').R[0])

        norm_parms = self.pack_norm_params(kwargs)
        for i, t in enumerate(progress_bar):
            x_t_input_h, x_t_input_o = self.get_image_conditioning(camera, image_rgb,
                                                                   kwargs, mask,
                                                                   norm_parms,
                                                                   t,
                                                                   xt_h, xt_o)

            # One reverse step with conditioning
            xt_h, xt_o = self.reverse_step(extra_step_kwargs, scheduler, t, torch.stack([xt_h, xt_o], 0),
                                    torch.stack([x_t_input_h, x_t_input_o], 0), **kwargs)  # (B, N, D), D=3

            if (return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1)):
                # print(xt_h.shape, kwargs.get('cent_hum').shape, kwargs.get('radius_hum').shape)
                x_t = torch.cat([self.denormalize_pclouds(xt_h, kwargs.get('cent_hum'), kwargs.get('radius_hum')),
                                 self.denormalize_pclouds(xt_o, kwargs.get('cent_obj'), kwargs.get('radius_obj'))], 1)
                # print(x_t.shape, xt_o.shape)
                all_outputs.append(torch.cat([x_t, segm_mask], -1))
                # print("Updating intermediate...")

        # Convert output back into a point cloud, undoing normalization and scaling
        x_t = torch.cat([self.denormalize_pclouds(xt_h, kwargs.get('cent_hum'), kwargs.get('radius_hum')),
                         self.denormalize_pclouds(xt_o, kwargs.get('cent_obj'), kwargs.get('radius_obj'))], 1)
        x_t = torch.cat([x_t, segm_mask], -1)
        output = self.tensor_to_point_cloud(x_t, denormalize=False, unscale=False)  # this convert the points back to original scale
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [self.tensor_to_point_cloud(o, denormalize=False, unscale=False) for o in all_outputs]

        return (output, all_outputs) if return_all_outputs else output

    def get_reverse_timesteps(self, scheduler, interm_steps: int):
        """
        get the timesteps to run reverse diffusion
        :param scheduler:
        :param interm_steps: start from some intermediate steps, the step number is for DDPM scheduler
            if DDIM, will be recomputed accordingly
        :return:
        """
        if isinstance(scheduler, DDPMScheduler):
            # DDPM, directly reverse N steps from interm_steps
            if interm_steps > 0:
                timesteps = torch.from_numpy(np.arange(0, interm_steps)[::-1].copy()).to(self.device)
            else:
                timesteps = scheduler.timesteps.to(self.device)
        elif isinstance(scheduler, DDIMScheduler):
            if interm_steps > 0:
                # compute a step ratio, and find the intermediate steps for DDIM
                step_ratio = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
                timesteps = (np.arange(0, interm_steps, step_ratio)).round()[::-1].copy().astype(np.int64)
                timesteps = torch.from_numpy(timesteps).to(self.device)
            else:
                timesteps = scheduler.timesteps.to(self.device)
        else:
            raise NotImplementedError
        return timesteps

    def pack_norm_params(self, kwargs:dict, scale=True):
        scale_factor = self.scale_factor if scale else 1.0
        hum = torch.cat([kwargs.get('cent_hum')*scale_factor, kwargs.get('radius_hum')], -1)
        obj = torch.cat([kwargs.get('cent_obj')*scale_factor, kwargs.get('radius_obj')], -1)
        return torch.stack([hum, obj], 0) # (2, B, 4)

    def reverse_step(self, extra_step_kwargs, scheduler, t, x_t, x_t_input, **kwargs):
        "x_t: (2, B, D, N), x_t_input: (2, B, D, N)"
        norm_parms = self.pack_norm_params(kwargs) # (2, B, 4)
        B = x_t.shape[1]
        # print(f"Step {t} Norm params:", norm_parms[:, 0, :])
        noise_pred_h, noise_pred_o = self.point_cloud_model(x_t_input[0], x_t_input[1], t.reshape(1).expand(B),
                                                            norm_parms)
        if self.consistent_center:
            assert self.dm_pred_type != 'sample', 'incompatible dm predition type!'
            noise_pred_h = noise_pred_h - torch.mean(noise_pred_h, dim=1, keepdim=True)
            noise_pred_o = noise_pred_o - torch.mean(noise_pred_o, dim=1, keepdim=True)

        xt_h = scheduler.step(noise_pred_h, t, x_t[0], **extra_step_kwargs).prev_sample
        xt_o = scheduler.step(noise_pred_o, t, x_t[1], **extra_step_kwargs).prev_sample

        if self.consistent_center:
            xt_h = xt_h - torch.mean(xt_h, dim=1, keepdim=True)
            xt_o = xt_o - torch.mean(xt_o, dim=1, keepdim=True)

        return xt_h, xt_o

    def denormalize_pclouds(self, x: Tensor, cent, radius, unscale: bool = True):
        """
        first denormalize, then apply center and scale to original H+O coordinate
        :param x:
        :param cent: (B, 3)
        :param radius: (B, 1)
        :param unscale:
        :return:
        """
        # denormalize: scale down.
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        # translation and scale back to H+O coordinate
        points = points * 2 * radius.unsqueeze(-1) + cent.unsqueeze(1)
        return points

    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        """
        take binary into account
        :param self:
        :param x: (B, N, 4)
        :param denormalize:
        :param unscale:
        :return:
        """
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        if self.predict_color:
            colors = self.denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
            return Pointclouds(points=points, features=colors)
        else:
            assert x.shape[2] == 4
            # add color to predicted binary labels
            is_hum = x[:, :, 3] > 0.5
            features = []
            for mask in is_hum:
                color = torch.zeros_like(x[0, :, :3]) + torch.tensor([0.5, 1.0, 0]).to(x.device)
                color[mask, :] = torch.tensor([0.05, 1.0, 1.0]).to(x.device)  # human is light blue, object light green
                features.append(color)
            return Pointclouds(points=points, features=features)


