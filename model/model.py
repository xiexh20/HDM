import inspect
import random
from typing import Optional

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

from .model_utils import get_num_points, get_custom_betas
from .point_cloud_model import PointCloudModel
from .projection_model import PointCloudProjectionModel


class ConditionalPointCloudDiffusionModel(PointCloudProjectionModel):
    
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model: str,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)
        
        # Checks
        if not self.predict_shape:
            raise NotImplementedError('Must predict shape if performing diffusion.')

        # Create diffusion model schedulers which define the sampling timesteps
        self.dm_pred_type = kwargs.get('dm_pred_type', "epsilon")
        assert self.dm_pred_type in ['epsilon','sample']
        scheduler_kwargs = {"prediction_type": self.dm_pred_type}
        if beta_schedule == 'custom':
            scheduler_kwargs.update(dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end)))
        else:
            scheduler_kwargs.update(dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule))
        self.schedulers_map = {
            'ddpm': DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            'ddim': DDIMScheduler(**scheduler_kwargs, clip_sample=False), 
            'pndm': PNDMScheduler(**scheduler_kwargs), 
        }
        self.scheduler = self.schedulers_map['ddpm']  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.init_pcloud_model(kwargs, point_cloud_model, point_cloud_model_embed_dim)

        self.load_sample_init = kwargs.get('load_sample_init', False)
        self.sample_init_scale = kwargs.get('sample_init_scale', 1.0)
        self.test_init_with_gtpc = kwargs.get('test_init_with_gtpc', False)

        self.consistent_center = kwargs.get('consistent_center', False)
        self.cam_noise_std = kwargs.get('cam_noise_std', 0.0) # add noise to camera based on timestamps

    def init_pcloud_model(self, kwargs, point_cloud_model, point_cloud_model_embed_dim):
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,  # voxel resolution multiplier is 1.
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

        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=True) # this will not pack the point colors
        B, N, D = x_0.shape

        # Sample random noise
        noise = torch.randn_like(x_0)
        if self.consistent_center:
            # modification suggested by https://arxiv.org/pdf/2308.07837.pdf
            noise = noise - torch.mean(noise, dim=1, keepdim=True)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), 
            device=self.device, dtype=torch.long)

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep) # diffusion noisy adding, only add to the coordinate, not features

        # add noise to the camera pose, based on timestamps
        if self.cam_noise_std > 0.000001:
            # the noise is very different
            camera = camera.clone()
            camT = camera.T # (B, 3)
            dist = torch.sqrt(torch.sum(camT**2, -1, keepdim=True))
            nratio = timestep[:, None] / self.scheduler.num_train_timesteps # time-dependent noise
            tnoise = torch.randn(B, 3).to(dist.device)/3. * dist * self.cam_noise_std * nratio
            camera.T = camera.T + tnoise

        # Conditioning, the pixel-aligned feature is based on points with noise (new points)
        x_t_input = self.get_diffu_input(camera, image_rgb, mask, timestep, x_t, **kwargs)

        # Forward
        loss, noise_pred = self.compute_loss(noise, timestep, x_0, x_t_input)

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss

    def compute_loss(self, noise, timestep, x_0, x_t_input):
        x_pred = torch.zeros_like(x_0)
        if self.self_conditioning:
            # self conditioning, from https://openreview.net/pdf?id=3itjR9QxFw
            if random.uniform(0, 1.) > 0.5:
                with torch.no_grad():
                    x_pred = self.point_cloud_model(torch.cat([x_t_input, x_pred], -1), timestep)
            noise_pred = self.point_cloud_model(torch.cat([x_t_input, x_pred], -1), timestep)
        else:
            noise_pred = self.point_cloud_model(x_t_input, timestep)
        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f'{noise_pred.shape=} and {noise.shape=}')
        # Loss
        if self.dm_pred_type == 'epsilon':
            loss = F.mse_loss(noise_pred, noise)
        elif self.dm_pred_type == 'sample':
            loss = F.mse_loss(noise_pred, x_0)  # predicting sample
        else:
            raise NotImplementedError
        return loss, noise_pred

    def get_diffu_input(self, camera, image_rgb, mask, timestep, x_t, **kwargs):
        "return: (B, N, D), the exact input to the diffusion model, x_t: (B, N, 3)"
        x_t_input = self.get_input_with_conditioning(x_t, camera=camera,
                                                     image_rgb=image_rgb, mask=mask, t=timestep)
        return x_t_input

    @torch.no_grad()
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

        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # Get the size of the noise
        N = num_points
        B = 1 if image_rgb is None else image_rgb.shape[0]
        D = self.get_x_T_channel()
        device = self.device if image_rgb is None else image_rgb.device

        sample_from_interm = kwargs.get('sample_from_interm', False)
        interm_steps = kwargs.get('noise_step') if sample_from_interm else -1
        x_t = self.initialize_x_T(device, gt_pc, (B, N, D), interm_steps, scheduler)
        x_pred = torch.zeros_like(x_t)

        # Set timesteps
        extra_step_kwargs = self.setup_reverse_process(eta, num_inference_steps, scheduler)

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f'Sampling ({x_t.shape})', disable=disable_tqdm)

        for i, t in enumerate(progress_bar):
            add_interm_output = (return_all_outputs and (
                        i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1))
            # Conditioning
            x_t_input = self.get_diffu_input(camera, image_rgb, mask, t, x_t, **kwargs)
            if self.self_conditioning:
                x_t_input = torch.cat([x_t_input, x_pred], -1) # add self-conditioning
            inference_binary = (i == len(progress_bar) - 1) | add_interm_output
            # One reverse step with conditioning
            x_t = self.reverse_step(extra_step_kwargs, scheduler, t, x_t, x_t_input,
                                    inference_binary=inference_binary) # (B, N, D), D=3 or 4
            x_pred = x_t # for next iteration self conditioning

            # Append to output list if desired
            if add_interm_output:
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(x_t, denormalize=True, unscale=True) # this convert the points back to original scale
        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [self.tensor_to_point_cloud(o, denormalize=True, unscale=True) for o in all_outputs]

        return (output, all_outputs) if return_all_outputs else output

    def get_x_T_channel(self):
        D = 3 + (self.color_channels if self.predict_color else 0)
        return D

    def initialize_x_T(self, device, gt_pc, shape, interm_steps:int=-1, scheduler=None):
        B, N, D = shape
        # Sample noise initialization
        if interm_steps > 0:
            # Sample from some intermediate steps
            x_0 = self.point_cloud_to_tensor(gt_pc, normalize=True, scale=True)
            noise = torch.randn(B, N, D, device=device)

            # always make sure the noise does not change the pc center, this is important to reduce 0.1cm CD!
            noise = noise - torch.mean(noise, dim=1, keepdim=True)

            x_t = scheduler.add_noise(x_0, noise, torch.tensor([interm_steps - 1] * B).long().to(device))  # Add noise
        else:
            # Sample from random Gaussian 
            x_t = torch.randn(B, N, D, device=device)

        x_t = x_t * self.sample_init_scale  # for test
        if self.consistent_center:
            x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)
        return x_t

    def reverse_step(self, extra_step_kwargs, scheduler, t, x_t, x_t_input, **kwargs):
        """
        run one reverse step to compute x_t
        :param extra_step_kwargs: 
        :param scheduler: 
        :param t: [1], diffusion time step
        :param x_t: (B, N, 3)
        :param x_t_input: conditional features (B, N, F)
        :param kwargs: other configurations to run diffusion step
        :return: denoised x_t
        """
        B = x_t.shape[0]
        # Forward
        noise_pred = self.point_cloud_model(x_t_input, t.reshape(1).expand(B))
        if self.consistent_center:
            assert self.dm_pred_type != 'sample', 'incompatible dm predition type for CCD!'
            # suggested by the CCD-3DR paper
            noise_pred = noise_pred - torch.mean(noise_pred, dim=1, keepdim=True)
        # Step
        x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample
        if self.consistent_center:
            x_t = x_t - torch.mean(x_t, dim=1, keepdim=True)
        return x_t

    def setup_reverse_process(self, eta, num_inference_steps, scheduler):
        """
        setup diffusion chain, and others.
        """
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}
        return extra_step_kwargs

    def forward(self, batch: FrameData, mode: str = 'train', **kwargs):
        """
        A wrapper around the forward method for training and inference
        """
        if isinstance(batch, dict):  # fixes a bug with multiprocessing where batch becomes a dict
            batch = FrameData(**batch)  # it really makes no sense, I do not understand it

        if mode == 'train':
            return self.forward_train(
                pc=batch.sequence_point_cloud, 
                camera=batch.camera,
                image_rgb=batch.image_rgb,
                mask=batch.fg_probability,
                **kwargs) 
        elif mode == 'sample':
            num_points = kwargs.pop('num_points', get_num_points(batch.sequence_point_cloud))
            return self.forward_sample(
                num_points=num_points,
                camera=batch.camera,
                image_rgb=batch.image_rgb,
                mask=batch.fg_probability,
                **kwargs)
        else:
            raise NotImplementedError()