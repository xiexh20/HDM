"""
Implementation of the ShapeNet dataloader to train pc2

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import numpy as np
import torch
import trimesh
import torchvision.transforms.functional as fn
from typing import Optional, List, Dict
import os.path as osp

from pytorch3d.datasets import R2N2, collate_batched_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.datasets.r2n2.utils import BlenderCamera


class R2N2Sample(R2N2):
    def __init__(self, num_samples, fix_sample=False, image_size=137,
                 augm_cfg=None,
                 split=None,
                 *params, **kwargs):
        super(R2N2Sample, self).__init__(split, *params, **kwargs)
        self.num_samples = num_samples
        self.fix_sample = fix_sample # same sample during training
        self.sample_buffers = {} # in case do deterministic sample
        self.image_size = image_size
        print("Num samples=", self.num_samples, ' fix_sample=', fix_sample, 'input image size=', image_size)
        self.split = split

        self.augm_cfg = augm_cfg # augmentation configuration
        print(self.augm_cfg)

    def __getitem__(self, model_idx, view_idxs: Optional[List[int]] = None) -> Dict:
        """
        load data, and then sample accordingly
        Parameters
        ----------
        model_idx
        view_idxs

        Returns
        -------

        """
        # print(model_idx)
        data_dict = super(R2N2Sample, self).__getitem__(model_idx, view_idxs)

        n_views = data_dict['images'].shape[0]
        if self.split == 'train':
            view_id = np.random.randint(0, n_views)
        else:
            view_id = 0 # always use one view for test

        if isinstance(model_idx, tuple):
            print("warning, found tuple:", model_idx)
            model_idx = model_idx[0]

        if self.fix_sample and model_idx in self.sample_buffers:
            p = self.sample_buffers[model_idx]
            print(f"Reusing previously sampled points for {model_idx}")
        else:
            # print(model_idx) no normalization was applied to the vertices, but it was already normalized?
            mesh = trimesh.Trimesh(data_dict['verts'].cpu().numpy(), data_dict['faces'].cpu().numpy(), process=False)
            p = torch.from_numpy(mesh.sample(self.num_samples)).float()
            self.sample_buffers[model_idx] = p
            # just check if it is normalized: most radius are between 0.47-0.498, yes, it is normalized!
            # radius = np.max(np.sqrt(np.sum(p.numpy()**2, -1)))
            # print(f"Mesh radius {model_idx} = {radius:.4f}")
        data_dict['pclouds'] = p

        if data_dict['images'].shape[1] != self.image_size:
            rgb_input = fn.resize(data_dict['images'][view_id].permute(2, 0, 1), [self.image_size, self.image_size])  # (3, H, W)
        else:
            rgb_input = data_dict['images'][view_id].permute(2, 0, 1)
        m = torch.mean(rgb_input, 0, keepdim=True) > 0

        # augmentation
        rgb_input, m = self.augm_images(rgb_input, m)

        # reshaped mask and images
        data_dict['images'] = rgb_input # (B, 3, H, W)
        data_dict['masks'] = m # (B, 1, H, W)

        data_dict['R'] = data_dict['R'][view_id]
        data_dict['T'] = data_dict['T'][view_id]
        data_dict['K'] = data_dict['K'][view_id]

        # delete textures
        if 'textures' in data_dict:
            data_dict.pop('textures') # don't load any texture, otherwise collate function will not work

        # additional information
        data_dict['image_path'] = osp.join(self.r2n2_dir, self.views_rel_path, data_dict['synset_id'], data_dict['model_id'], f'rendering/{view_id:02d}.png')
        data_dict['sequence_name'] = f"{data_dict['model_id']}"
        data_dict['view_id'] = view_id
        data_dict['image_size_hw'] = torch.tensor([self.image_size, self.image_size])

        return data_dict

    def augm_images(self, rgb, mask):
        """
        do data augmentation, mainly adding occlusion to the images
        Parameters
        ----------
        rgb (3, H, W) tensor, float
        mask (1, H, W) tensor boolean

        Returns
        -------
        same rgb and mask, with some randomly generated regions being masked out
        """
        if self.augm_cfg is None or self.augm_cfg.max_radius == 0:
            return rgb, mask # no augmentation

        iy, ix = torch.where(mask[0]>0) # the indices for mask pixels

        if len(iy) < 10:
            return rgb, mask # no augmentation

        # pick a random center
        ind = np.random.randint(0, len(iy))
        # generate a random square size
        size = np.random.randint(1, self.augm_cfg.max_radius+1)
        # size = self.augm_cfg.max_radius # for debug
        h, w = mask.shape[1:]

        # get the square coordinates
        x1 = max(0, ix[ind] - size)
        x2 = min(w, ix[ind] + size)
        y1 = max(0, iy[ind] - size)
        y2 = min(h, iy[ind] + size)

        # now set this region to zero
        rgb[:, y1:y2, x1:x2] = 0
        mask[:, y1:y2, x1:x2] = False

        return rgb, mask

