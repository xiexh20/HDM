from typing import Optional, Union

import torch
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from diffusers import ModelMixin
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer import PointsRasterizationSettings, PointsRasterizer
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor

from .feature_model import FeatureModel
from .model_utils import compute_distance_transform

SchedulerClass = Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]


class PointCloudProjectionModel(ModelMixin):
    
    def __init__(
        self,
        image_size: int,
        image_feature_model: str,
        use_local_colors: bool = True,
        use_local_features: bool = True,
        use_global_features: bool = False,
        use_mask: bool = True,
        use_distance_transform: bool = True,
        predict_shape: bool = True,
        predict_color: bool = False,
        process_color: bool = False,
        image_color_channels: int = 3,  # for the input image, not the points
        color_channels: int = 3,  # for the points, not the input image
        colors_mean: float = 0.5,
        colors_std: float = 0.5,
        scale_factor: float = 1.0,
        # Rasterization settings
        raster_point_radius: float = 0.0075,  # point size
        raster_points_per_pixel: int = 1,  # a single point per pixel, for now
        bin_size: int = 0,
            model_name=None,
            # additional arguments added by XH
            load_sample_init=False,
            sample_init_scale=1.0,
            test_init_with_gtpc=False,
            consistent_center=False, # from https://arxiv.org/pdf/2308.07837.pdf
            voxel_resolution_multiplier: int=1,
            predict_binary: bool=False, # predict a binary class label
            lw_binary: float=1.0,
            binary_training_noise_std: float=0.1,
            dm_pred_type: str='epsilon', # diffusion prediction type
            self_conditioning=False,
            **kwargs,

    ):
        super().__init__()
        self.image_size = image_size
        self.scale_factor = scale_factor
        self.use_local_colors = use_local_colors
        self.use_local_features = use_local_features
        self.use_global_features = use_global_features
        self.use_mask = use_mask
        self.use_distance_transform = use_distance_transform
        self.predict_shape = predict_shape # default False
        self.predict_color = predict_color # default True
        self.process_color = process_color
        self.image_color_channels = image_color_channels
        self.color_channels = color_channels
        self.colors_mean = colors_mean
        self.colors_std = colors_std
        self.model_name = model_name
        print("PointCloud Model scale factor:", self.scale_factor, 'Model name:', self.model_name)
        self.predict_binary = predict_binary
        self.lw_binary = lw_binary
        self.self_conditioning = self_conditioning

        # Types of conditioning that are used
        self.use_local_conditioning = self.use_local_colors or self.use_local_features or self.use_mask
        self.use_global_conditioning = self.use_global_features
        self.kwargs = kwargs

        # Create feature model
        self.feature_model = FeatureModel(image_size, image_feature_model)

        # Input size
        self.in_channels = 3  # 3 for 3D point positions
        if self.use_local_colors: # whether color should be an input
            self.in_channels += self.image_color_channels
        if self.use_local_features:
            self.in_channels += self.feature_model.feature_dim
        if self.use_global_features:
            self.in_channels += self.feature_model.feature_dim
        if self.use_mask:
            self.in_channels += 2 if self.use_distance_transform else 1
        if self.process_color:
            self.in_channels += self.color_channels # point color added to input or not, default False
        if self.self_conditioning:
            self.in_channels += 3 # add self conditioning

        self.in_channels = self.add_extra_input_chennels(self.in_channels)

        if self.model_name in ['pc2-diff-ho-sepsegm', 'diff-ho-attn']:
            self.in_channels += 2 if self.use_distance_transform else 1

        # Output size
        self.out_channels = 0
        if self.predict_shape:
            self.out_channels += 3
        if self.predict_color:
            self.out_channels += self.color_channels
        if self.predict_binary:
            print("Output binary classification score!")
            self.out_channels += 1

        # Save rasterization settings
        self.raster_settings = PointsRasterizationSettings(
            image_size=(image_size, image_size),
            radius=raster_point_radius,
            points_per_pixel=raster_points_per_pixel,
            bin_size=bin_size,
        )

    def add_extra_input_chennels(self, input_channels):
        return input_channels
    
    def denormalize(self, x: Tensor, /, clamp: bool = True):
        x = x * self.colors_std + self.colors_mean
        return torch.clamp(x, 0, 1) if clamp else x

    def normalize(self, x: Tensor, /):
        x = (x - self.colors_mean) / self.colors_std
        return x

    def get_global_conditioning(self, image_rgb: Tensor):
        global_conditioning = []
        if self.use_global_features:
            global_conditioning.append(self.feature_model(image_rgb, 
                return_cls_token_only=True))  # (B, D)
        global_conditioning = torch.cat(global_conditioning, dim=1)  # (B, D_cond)
        return global_conditioning

    def get_local_conditioning(self, image_rgb: Tensor, mask: Tensor):
        """
        compute per-point conditioning
        Parameters
        ----------
        image_rgb: (B, 3, 224, 224), values normalized to 0-1, background is masked by the given mask
        mask: (B, 1, 224, 224), or (B, 2, 224, 224) for h+o
        """
        local_conditioning = []
        # import pdb; pdb.set_trace()

        if self.use_local_colors: # XH: default True
            local_conditioning.append(self.normalize(image_rgb))
        if self.use_local_features: # XH: default True
            local_conditioning.append(self.feature_model(image_rgb)) # I guess no mask here? feature model: 'vit_small_patch16_224_mae'
        if self.use_mask: # default True
            local_conditioning.append(mask.float())
        if self.use_distance_transform: # default True
            if not self.use_mask:
                raise ValueError('No mask for distance transform?')
            if mask.is_floating_point():
                mask = mask > 0.5
            local_conditioning.append(compute_distance_transform(mask))
        local_conditioning = torch.cat(local_conditioning, dim=1)  # (B, D_cond, H, W)
        return local_conditioning

    @torch.autocast('cuda', dtype=torch.float32)
    def surface_projection(
        self, points: Tensor, camera: CamerasBase, local_features: Tensor,
    ):        
        B, C, H, W, device = *local_features.shape, local_features.device
        R = self.raster_settings.points_per_pixel
        N = points.shape[1]
        
        # Scale camera by scaling T. ASSUMES CAMERA IS LOOKING AT ORIGIN!
        camera = camera.clone()
        camera.T = camera.T * self.scale_factor

        # Create rasterizer
        rasterizer = PointsRasterizer(cameras=camera, raster_settings=self.raster_settings)

        # Associate points with features via rasterization
        fragments = rasterizer(Pointclouds(points))  # (B, H, W, R)
        fragments_idx: Tensor = fragments.idx.long()
        visible_pixels = (fragments_idx > -1)  # (B, H, W, R)
        points_to_visible_pixels = fragments_idx[visible_pixels]

        # Reshape local features to (B, H, W, R, C)
        local_features = local_features.permute(0, 2, 3, 1).unsqueeze(-2).expand(-1, -1, -1, R, -1)  # (B, H, W, R, C)

        # Get local features corresponding to visible points
        local_features_proj = torch.zeros(B * N, C, device=device)
        # local feature includes: raw RGB color, image features, mask, distance transform
        local_features_proj[points_to_visible_pixels] = local_features[visible_pixels]
        local_features_proj = local_features_proj.reshape(B, N, C)
        
        return local_features_proj
    
    def point_cloud_to_tensor(self, pc: Pointclouds, /, normalize: bool = False, scale: bool = False):
        """Converts a point cloud to a tensor, with color if and only if self.predict_color"""
        points = pc.points_padded() * (self.scale_factor if scale else 1)
        if self.predict_color and pc.features_padded() is not None: # normalize color, not point locations
            colors = self.normalize(pc.features_padded()) if normalize else pc.features_padded()
            return torch.cat((points, colors), dim=2)
        else:
            return points
    
    def tensor_to_point_cloud(self, x: Tensor, /, denormalize: bool = False, unscale: bool = False):
        points = x[:, :, :3] / (self.scale_factor if unscale else 1)
        if self.predict_color:
            colors = self.denormalize(x[:, :, 3:]) if denormalize else x[:, :, 3:]
            return Pointclouds(points=points, features=colors)
        else:
            assert x.shape[2] == 3
            return Pointclouds(points=points)

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        t: Optional[Tensor],
    ):
        """ Extracts local features from the input image and projects them onto the points 
            in the point cloud to obtain the input to the model. Then extracts global 
            features, replicates them across points, and concats them to the input.
            image_rgb: masked background
            XH: why there is no positional encoding as described by the supp??
            """
        B, N = x_t.shape[:2]
        
        # Initial input is the point locations (and colors if and only if predicting color)
        x_t_input = self.get_coord_feature(x_t)

        # Local conditioning
        if self.use_local_conditioning:

            # Get local features and check that they are the same size as the input image
            local_features = self.get_local_conditioning(image_rgb=image_rgb, mask=mask) # concatenate RGB + mask + RGB feature + distance transform
            if local_features.shape[-2:] != image_rgb.shape[-2:]:
                raise ValueError(f'{local_features.shape=} and {image_rgb.shape=}')
            
            # Project local features. Here that we only need the point locations, not colors
            local_features_proj = self.surface_projection(points=x_t[:, :, :3],
                camera=camera, local_features=local_features)  # (B, N, D_local)

            x_t_input.append(local_features_proj)

        # Global conditioning
        if self.use_global_conditioning: # False

            # Get and repeat global features
            global_features = self.get_global_conditioning(image_rgb=image_rgb)  # (B, D_global)
            global_features = global_features.unsqueeze(1).expand(-1, N, -1)  # (B, D_global, N)

            x_t_input.append(global_features)

        # Concatenate together all the pointwise features
        x_t_input = torch.cat(x_t_input, dim=2)  # (B, N, D)

        return x_t_input

    def get_coord_feature(self, x_t):
        """get coordinate feature, for model that uses separate model to predict binary, we use first 3 channels only"""
        x_t_input = [x_t]
        return x_t_input

    def forward(self, batch: FrameData, mode: str = 'train', **kwargs):
        """ The forward method may be defined differently for different models. """
        raise NotImplementedError()
