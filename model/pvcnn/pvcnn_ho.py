"""
two separate diffusion models for human+object, with cross attention to communicate between them
"""
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import math

from model.pvcnn.modules import Attention, PVConv, BallQueryHO
from model.pvcnn.pvcnn_utils import create_mlp_components, create_pointnet2_sa_components, create_pointnet2_fp_modules
from model.pvcnn.pvcnn_utils import get_timestep_embedding
import torch.nn.functional as F
from .pos_enc import get_embedder


def _scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.

    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.

    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.

        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    if attn_mask is not None:
        attn = torch.baddbmm(attn_mask, q, k.transpose(-2, -1))
    else:
        attn = torch.bmm(q, k.transpose(-2, -1))

    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p) # this is only for training?
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


class PVCNN2HumObj(nn.Module):
    sa_blocks = [
        # (out_channel, num_blocks, voxel_reso), (num_centers, radius, num_neighbors, out_channels)
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        # (out, in_channels), (out_channels, num_blocks, voxel_resolution)
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
            self,
            num_classes: int,
            embed_dim: int,
            use_att: bool = True,
            dropout: float = 0.1,
            extra_feature_channels: int = 3,
            width_multiplier: int = 1,
            voxel_resolution_multiplier: int = 1,
            attn_type: str = 'simple-cross', #
            attn_weight: float=1.0, # attention feature weight
            multires: int = 10, # positional encoding resolution
            num_neighbours: int = 32 # ball query neighbours
    ):
        super(PVCNN2HumObj, self).__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.width_multiplier = width_multiplier
        self.num_neighbours = num_neighbours
        self.in_channels = extra_feature_channels + 3

        self.attn_type = attn_type # how to compute attention
        self.attn_weight = attn_weight

        # separate human/object model
        classifier, embedf, fp_layers, global_att, sa_layers = self.make_modules(dropout, embed_dim,
                                                                                 extra_feature_channels, num_classes,
                                                                                 use_att, voxel_resolution_multiplier,
                                                                                 width_multiplier)

        self.sa_layers_hum = sa_layers
        self.global_att_hum = global_att
        self.fp_layers_hum = fp_layers
        self.classifier_hum = classifier
        self.embedf_hum = embedf
        self.posi_encoder, _ = get_embedder(multires)

        classifier, embedf, fp_layers, global_att, sa_layers = self.make_modules(dropout, embed_dim,
                                                                                 extra_feature_channels, num_classes,
                                                                                 use_att, voxel_resolution_multiplier,
                                                                                 width_multiplier)

        self.sa_layers_obj = sa_layers
        self.global_att_obj = global_att
        self.fp_layers_obj = fp_layers
        self.classifier_obj = classifier
        self.embedf_obj = embedf

        self.make_coord_attn()
        assert self.attn_type == 'coord3d+posenc-learnable', f'unknown attention type {self.attn_type}'

    def make_modules(self, dropout, embed_dim, extra_feature_channels, num_classes, use_att,
                     voxel_resolution_multiplier, width_multiplier):
        """
        make module for human/object
        :param dropout:
        :param embed_dim:
        :param extra_feature_channels:
        :param num_classes:
        :param use_att:
        :param voxel_resolution_multiplier:
        :param width_multiplier:
        :return:
        """
        in_ch_multiplier = 1
        extra_in_channel = 63  # the segmentation+positional feature is projected to dim 63

        # Create PointNet-2 model
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks_config=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            in_ch_multiplier=in_ch_multiplier,
            extra_in_channel=extra_in_channel
        )
        sa_layers = nn.ModuleList(sa_layers)
        # Additional global attention module, default true
        if self.attn_type == 'coord3d+posenc+rgb':
            # reduce channel number, only for the global attention layer, the decoders remain unchanged
            global_att = None if not use_att else Attention(channels_sa_features//2, 8, D=1)
        else:
            global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            in_ch_multiplier=in_ch_multiplier,
            extra_in_channel=extra_in_channel
        )
        fp_layers = nn.ModuleList(fp_layers)

        # Create MLP layers for output prediction
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier
        )
        classifier = nn.Sequential(*layers)  # applied to point features directly
        # Time embedding function
        embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        return classifier, embedf, fp_layers, global_att, sa_layers

    def make_coord_attn(self):
        "learnable attention only on point coordinate + positional encoding "
        pvconv_encoders = []
        for i, (conv_configs, sa_configs) in enumerate(self.sa_blocks):
            # should use point net out channel
            out_channel = 63
            layer = nn.MultiheadAttention(out_channel, 1, batch_first=True, kdim=out_channel, vdim=out_channel+2)
            pvconv_encoders.append(layer)  # only one block for conv
        pvconv_decoders = []
        for fp_configs, conv_configs in self.fp_blocks:
            out_channel = 63
            layer = nn.MultiheadAttention(out_channel, 1, batch_first=True, kdim=out_channel, vdim=out_channel + 2)
            pvconv_decoders.append(layer)
        self.cross_conv_encoders = nn.ModuleList(pvconv_encoders)
        self.cross_conv_decoders = nn.ModuleList(pvconv_decoders)

    def forward(self, inputs_hum: torch.Tensor, inputs_obj: torch.Tensor, t: torch.Tensor, norm_params=None):
        """

        :param inputs: (B, N, D), N is the number of points, D is the conditional feature dimension
        :param t: (B, ) timestamps
        :param norm_params: (2, B, 4), transformation parameters that move points back to H+O joint space, first 3 values are cent, the last is radius/scale
        :return: (B, N, D_out) x2
        """
        inputs_hum = inputs_hum.transpose(1, 2)
        inputs_obj = inputs_obj.transpose(1, 2)

        # Embed timesteps, sinusoidal encoding
        t_emb_init = get_timestep_embedding(self.embed_dim, t, inputs_hum.device).float()
        t_emb_hum = self.embedf_hum(t_emb_init)[:, :, None].expand(-1, -1, inputs_hum.shape[-1]).float()
        t_emb_obj = self.embedf_obj(t_emb_init)[:, :, None].expand(-1, -1, inputs_obj.shape[-1]).float()

        # Separate input coordinates and features
        coords_hum, coords_obj = inputs_hum[:, :3, :].contiguous(), inputs_obj[:, :3, :].contiguous()  # (B, 3, N) range (-3.5, 3.5)
        features_hum, features_obj = inputs_hum, inputs_obj  # (B, 3 + S, N)

        DEBUG = False

        # Encoder: Downscaling layers
        coords_list_hum, coords_list_obj = [], []
        in_features_list_hum, in_features_list_obj = [], []
        for i, (sa_blocks_h, sa_blocks_o) in enumerate(zip(self.sa_layers_hum, self.sa_layers_obj)):
            in_features_list_hum.append(features_hum)
            coords_list_hum.append(coords_hum)
            in_features_list_obj.append(features_obj)
            coords_list_obj.append(coords_obj)
            if i == 0:
                # First step no timestamp embedding
                features_hum, coords_hum, t_emb_hum = sa_blocks_h((features_hum, coords_hum, t_emb_hum))
                features_obj, coords_obj, t_emb_obj = sa_blocks_o((features_obj, coords_obj, t_emb_obj))
            else:
                features_hum, coords_hum, t_emb_hum = sa_blocks_h((torch.cat([features_hum, t_emb_hum], dim=1), coords_hum, t_emb_hum))
                features_obj, coords_obj, t_emb_obj = sa_blocks_o((torch.cat([features_obj, t_emb_obj], dim=1), coords_obj, t_emb_obj))

            if i < len(self.sa_layers_hum)-1:
                features_hum, features_obj = self.add_attn_feature(features_hum, features_obj,
                                                               self.transform_coords(coords_hum, norm_params, 0),
                                                               self.transform_coords(coords_obj, norm_params, 1),
                                                               self.cross_conv_encoders[i],
                                                                   temb_hum=t_emb_hum,
                                                                   temb_obj=t_emb_obj)

        # for debug: save some point clouds
        if DEBUG:
            for i, (ch, co) in enumerate(zip(coords_list_hum, coords_list_obj)):
                import trimesh
                ch_ho = self.transform_coords(ch, norm_params, 0)
                co_ho = self.transform_coords(co, norm_params, 1)
                points = torch.cat([ch_ho, co_ho], -1).transpose(1, 2)
                L = ch_ho.shape[-1]
                vc = np.concatenate(
                    [np.zeros((L, 3)) + np.array([0.5, 1.0, 0]),
                     np.zeros((L, 3)) + np.array([0.05, 1.0, 1.0])]
                )
                trimesh.PointCloud(points[0].cpu().numpy(), colors=vc).export(
                    f'/BS/xxie-2/work/pc2-diff/experiments/debug/meshes/encoder_step{i:02d}.ply')

        # Replace the input features
        in_features_list_hum[0] = inputs_hum[:, 3:, :].contiguous()
        in_features_list_obj[0] = inputs_obj[:, 3:, :].contiguous()

        # Apply global attention layer
        if self.global_att_hum is not None:
            features_hum = self.global_att_hum(features_hum)

        if self.global_att_obj is not None:
            features_obj = self.global_att_obj(features_obj)
        # Do cross attention after self-attention
        if self.attn_type in ['coord3d+posenc-learnable']:
            features_hum, features_obj = self.add_attn_feature(features_hum, features_obj,
                                                               self.transform_coords(coords_hum, norm_params, 0),
                                                               self.transform_coords(coords_obj, norm_params, 1),
                                                               self.cross_conv_encoders[-1] if self.attn_type in [
                                                                   'coord3d+posenc-learnable'] else None,
                                                               temb_hum=t_emb_hum,
                                                               temb_obj=t_emb_obj)

        # Upscaling layers
        for fp_idx, (fp_blocks_h, fp_blocks_o) in enumerate(zip(self.fp_layers_hum, self.fp_layers_obj)):
            features_hum, coords_hum, t_emb_hum = fp_blocks_h(
                (  # this is a tuple because of nn.Sequential
                    coords_list_hum[-1 - fp_idx],  # reverse coords list from above
                    coords_hum,  # original point coordinates
                    torch.cat([features_hum, t_emb_hum], dim=1),  # keep concatenating upsampled features with timesteps
                    in_features_list_hum[-1 - fp_idx],  # reverse features list from above
                    t_emb_hum  # original timestep embedding
                )
                # this is where point voxel convolution is carried out, the point feature network preserves the order.
            )
            features_obj, coords_obj, t_emb_obj = fp_blocks_o(
                (  # this is a tuple because of nn.Sequential
                    coords_list_obj[-1 - fp_idx],  # reverse coords list from above
                    coords_obj,  # original point coordinates
                    torch.cat([features_obj, t_emb_obj], dim=1),  # keep concatenating upsampled features with timesteps
                    in_features_list_obj[-1 - fp_idx],  # reverse features list from above
                    t_emb_obj  # original timestep embedding
                )
                # this is where point voxel convolution is carried out, the point feature network preserves the order.
            )

            # these features are reused as input for next layer
            # add attention except for the last layer
            if fp_idx < len(self.fp_layers_hum) - 1:
                # Perform cross attention between human and object branches
                features_hum, features_obj = self.add_attn_feature(features_hum, features_obj,
                                                                   self.transform_coords(coords_hum, norm_params, 0),
                                                                   self.transform_coords(coords_obj, norm_params, 1),
                                                                   self.cross_conv_decoders[fp_idx] if self.attn_type in ['coord3d+posenc-learnable'] else None,
                                                                   temb_hum=t_emb_hum,
                                                                   temb_obj=t_emb_obj
                                                                   )

            if DEBUG:
                import trimesh
                ch_ho = self.transform_coords(coords_hum, norm_params, 0)
                co_ho = self.transform_coords(coords_obj, norm_params, 1)
                points = torch.cat([ch_ho, co_ho], -1).transpose(1, 2)
                L = ch_ho.shape[-1]
                vc = np.concatenate(
                    [np.zeros((L, 3)) + np.array([0.5, 1.0, 0]),
                     np.zeros((L, 3)) + np.array([0.05, 1.0, 1.0])]
                )
                trimesh.PointCloud(points[0].cpu().numpy(), colors=vc).export(
                    f'/BS/xxie-2/work/pc2-diff/experiments/debug/meshes/decoder_step{fp_idx:02d}.ply')

        if DEBUG:
            exit(0)
        # Output MLP layers
        output_hum = self.classifier_hum(features_hum).transpose(1, 2) # convert back to (B, N, D) format
        output_obj = self.classifier_obj(features_obj).transpose(1, 2)

        return output_hum, output_obj

    def transform_coords(self, coords, norm_params, target_ind):
        """
        transform coordinates such that the points align back to H+O interaction space
        :param coords: (B, 3, N)
        :param norm_params: (2, B, 4)
        :param target_ind: 0 or 1
        :return:
        """
        scale = norm_params[target_ind, :, 3:].unsqueeze(1)
        cent = norm_params[target_ind, :, :3].unsqueeze(-1)
        coords_ho = coords * 2 * scale + cent
        return coords_ho

    def add_attn_feature(self, features_hum, features_obj,
                         coords_hum=None, coords_obj=None,
                         attn_module=None,
                         temb_hum=None, temb_obj=None):
        """
        compute cross attention between human and object points
        :param features_hum: (B, D, N)
        :param features_obj: (B, D, N)
        :param coords_hum: (B, 3, N), human points in the H+O frame
        :param coords_obj: (B, 3, N), object points in the H+O frame
        :param temb: time embedding
        :return: cross attended human object features.
        """
        B, D, N = features_hum.shape
        # the attn_module is learnable, only difference is the number of output feature dimension
        onehot_hum, onehot_obj = self.get_onehot_feat(features_hum)
        pos_hum = self.posi_encoder(coords_hum.permute(0, 2, 1)).permute(0, 2, 1)
        pos_obj = self.posi_encoder(coords_obj.permute(0, 2, 1)).permute(0, 2, 1)
        feat_hum = torch.cat([pos_hum, onehot_hum], 1)
        feat_obj = torch.cat([pos_obj, onehot_obj], 1)  # (B, 65, N)

        attn_h2o = attn_module(pos_obj.permute(0, 2, 1),
                               pos_hum.permute(0, 2, 1),
                               feat_hum.permute(0, 2, 1))[0].permute(0, 2, 1)
        attn_o2h = attn_module(pos_hum.permute(0, 2, 1),
                               pos_obj.permute(0, 2, 1),
                               feat_obj.permute(0, 2, 1))[0].permute(0, 2, 1)
        features_hum = torch.cat([features_hum, attn_o2h * self.attn_weight], 1)
        features_obj = torch.cat([features_obj, attn_h2o * self.attn_weight], 1)

        return features_hum, features_obj

    def get_onehot_feat(self, features_hum):
        """
        compute a onehot feature vector to identify this is human or object
        :param features_hum:
        :return: (B, 2, N) x2 for human and object
        """
        B, D, N = features_hum.shape
        onehot_hum = torch.zeros(B, 2, N).to(features_hum.device)
        onehot_hum[:, 0] = 1.
        onehot_obj = torch.zeros(B, 2, N).to(features_hum.device)
        onehot_obj[:, 1] = 1.0
        return onehot_hum, onehot_obj


