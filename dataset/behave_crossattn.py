"""
Dataset for stage2 model

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import os
import os.path as osp
import numpy as np
import cv2, trimesh
import torch
import igl
import pickle as pkl
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import time

from .behave_paths import DataPaths, date_seqs
from behave.kinect_transform import KinectTransform
from .utils import create_grid_points
from .behave_dataset import BehaveDataset
from .img_utils import compute_translation
from .img_utils import compute_translation, crop, masks2bbox, resize

class BehaveCrossAttnDataset(BehaveDataset):
    def get_item(self, idx):
        "sample human + object, compute K, cent, translation etc. for only human or object"
        rgb_file = self.data_paths[idx]
        mask_hum, mask_obj = self.load_masks(rgb_file)
        rgb_full = cv2.imread(rgb_file)[:, :, ::-1]
        color_h, color_w = rgb_full.shape[:2]
        if self.split == 'train' and np.random.uniform()>0.5:
            rgb_full = self.blur_image(rgb_full, self.aug_blur)

        # Samples
        samples_obj, samples_smpl = self.get_samples(rgb_file)
        cent_ho, radius_ho = self.get_normalize_params(np.concatenate([samples_smpl, samples_obj], 0), samples_smpl)
        samples_smpl = (samples_smpl - cent_ho)/ (2*radius_ho)
        samples_obj = (samples_obj - cent_ho)/(2*radius_ho)

        # Now normalize each sample to unit sphere
        cent_h, radius_h = self.unit_sphere_params(samples_smpl)
        cent_o, radius_o = self.unit_sphere_params(samples_obj)
        samples_smpl = (samples_smpl - cent_h) / (2*radius_h)
        samples_obj = (samples_obj - cent_o) / (2 * radius_o)
        # compute translation parameters for projection
        T_ho_scaled = cent_ho / (radius_ho * 2)
        T_hum_scaled = (T_ho_scaled + cent_h) / (2*radius_h) * np.array([-1, -1, 1.])
        T_obj_scaled = (T_ho_scaled + cent_o) / (2 * radius_o) * np.array([-1, -1, 1.])
        T_ho_scaled = T_ho_scaled * np.array([-1, -1, 1.])

        # compute an estimated center
        if self.test_transl_type == 'estimated':
            _, _, crop_center_ho, crop_size_ho = self.get_crop_params(mask_hum, mask_obj, 1.0)
            is_behave = self.is_behave_dataset(rgb_full.shape[1])
            transl_estimate = compute_translation(crop_center_ho, crop_size_ho, is_behave, self.std_coverage)
            T_ho_scaled = transl_estimate / 7.0 * np.array([-1, -1, 1.])
            radius_ho = 0.5
            cent_ho = transl_estimate / 7.0
            print("Using estimated center for H+O", transl_estimate.tolist())
        elif self.test_transl_type == 'estimated-2d':
            _, _, crop_center_ho, crop_size_ho = self.get_crop_params(mask_hum, mask_obj, 1.0)
            is_behave = self.is_behave_dataset(rgb_full.shape[1])
            assert rgb_full.shape[1] in [2048, 1920], 'the image is not normalized to BEHAVE or ICAP size!'
            indices = np.indices(rgb_full.shape[:2])
            assert np.sum(mask_obj > 127) > 5, f'not enough object mask found for {rgb_file}'
            pts_h = np.stack([indices[1][mask_hum > 127], indices[0][mask_hum > 127]], -1)
            pts_o = np.stack([indices[1][mask_obj > 127], indices[0][mask_obj > 127]], -1)
            proj_cent_est = (np.mean(pts_h, 0) + np.mean(pts_o, 0)) / 2.
            transl_estimate = compute_translation(proj_cent_est, crop_size_ho, is_behave, self.std_coverage)
            T_ho_scaled = transl_estimate / 7.0 * np.array([-1, -1, 1.])
            radius_ho = 0.5
            cent_ho = transl_estimate / 7.0
            print(f"Cross-attn dataset estimated 2d: {proj_cent_est}, 3D: {transl_estimate / 7.}")

        # crop for H+O
        Kroi, objmask_fullcrop, psmask_fullcrop, rgb_fullcrop = self.crop_full_image(mask_hum.copy(),
                                                                  mask_obj.copy(),
                                                                  rgb_full.copy(),
                                                                  [mask_hum, mask_obj],
                                                                                     1.00) # October 25: with small extension, this is different from segmentation input, it is not used anyway
        # crop human only
        if self.sep_same_crop:
            Kroi_h, obj_mask, person_mask, rgb = Kroi.copy(), objmask_fullcrop.copy(), psmask_fullcrop.copy(), rgb_fullcrop.copy()
        else:
            Kroi_h, obj_mask, person_mask, rgb = self.crop_full_image(mask_hum.copy(),
                                                                mask_obj.copy(),
                                                                rgb_full.copy(),
                                                                [mask_hum, mask_hum], 1.05)
        # print(f"{rgb_file} Camera T:", T_ho_scaled) # this one is quite different for behave data
        ss = rgb_file.split(os.sep)
        data_dict = {
            # parameter for h+o
            "R": torch.from_numpy(self.opencv2py3d[:3, :3]).float(),
            "T": torch.from_numpy(T_ho_scaled).float(),
            "K": torch.from_numpy(Kroi).float(),
            "images_fullcrop":torch.from_numpy(rgb_fullcrop).float().permute(2, 0, 1),
            "masks_fullcrop": torch.from_numpy(np.stack([psmask_fullcrop, objmask_fullcrop], 0)).float(),

            # human only
            "images": torch.from_numpy(rgb).float().permute(2, 0, 1),
            "masks": torch.from_numpy(np.stack([person_mask, obj_mask], 0)).float(),
            "K_hum": torch.from_numpy(Kroi_h).float(),
            "T_hum": torch.from_numpy(T_hum_scaled).float(),
            "pclouds": torch.from_numpy(samples_smpl).float(),

            # object only
            "T_obj": torch.from_numpy(T_obj_scaled).float(),
            "pclouds_obj": torch.from_numpy(samples_obj).float(),

            # additional information
            "image_path": rgb_file,
            'view_id': DataPaths.get_kinect_id(rgb_file),
            "sequence_name": ss[-2],  # the frame name
            "synset_id": ss[-3],

            # normalization parameters
            "cent_hum":torch.from_numpy(cent_h).float(),
            "cent_obj":torch.from_numpy(cent_o).float(),
            "radius_hum":torch.tensor([radius_h]).float(),
            "radius_obj": torch.tensor([radius_o]).float(),
            # H+O
            "gt_trans": cent_ho,
            'radius': radius_ho,

            "image_size_hw": torch.tensor(self.input_size), # for metadata

        }

        # object only image input
        # TODO: check if object mask is zero, if so, then put full crop of human + object to the object, or we compute a GT projection?
        # the idea is to enable more generative power for the object
        if self.sep_same_crop:
            Kroi_o, obj_mask, person_mask, rgb = Kroi.copy(), objmask_fullcrop.copy(), psmask_fullcrop.copy(), rgb_fullcrop.copy()
        else:
            try:
                Kroi_o, obj_mask, person_mask, rgb = self.crop_full_image(mask_hum.copy(),
                                                                      mask_obj.copy(),
                                                                      rgb_full.copy(),
                                                                      [mask_obj, mask_obj], 1.5)
            except Exception as e:
                print(f"Failed on {rgb_file} due to {e}")
                if np.sum(mask_obj > 127) < 10:
                    # TODO: understand how will this affect small objects
                    print("Using full crop to replace object only crop for", rgb_file)
                    # use full crop
                    Kroi_o, obj_mask, person_mask, rgb = Kroi.copy(), objmask_fullcrop.copy(), psmask_fullcrop.copy(), rgb_fullcrop.copy()
                else:
                    Kroi_o, obj_mask, person_mask, rgb = self.crop_full_image(mask_hum.copy(),
                                                                      mask_obj.copy(),
                                                                      rgb_full.copy(),
                                                                      [mask_obj, mask_obj], 1.5)

        data_dict = {
            **data_dict,
            "images_obj": torch.from_numpy(rgb).float().permute(2, 0, 1),
            "masks_obj":torch.from_numpy(np.stack([person_mask, obj_mask], 0)).float(),
            "K_obj":torch.from_numpy(Kroi_o).float()
        }

        if self.ho_segm_pred_path is not None:
            # load predicted H+O
            pred_dict = self.load_predicted_HO(cent_ho, radius_ho, rgb_file)
            data_dict = {
                **data_dict,

                **pred_dict
            }


        return data_dict

    def crop_full_image(self, mask_hum, mask_obj, rgb_full, crop_masks, bbox_exp=1.0):
        """
        crop the image based on the given masks
        :param mask_hum:
        :param mask_obj:
        :param rgb_full:
        :param crop_masks: a list of masks used to do the crop
        :return: Kroi, cropped human, object mask and RGB images (background masked out).
        """
        bmax, bmin, crop_center, crop_size = self.get_crop_params(*crop_masks, bbox_exp)
        rgb = resize(crop(rgb_full, crop_center, crop_size), self.input_size) / 255.
        person_mask = resize(crop(mask_hum, crop_center, crop_size), self.input_size) / 255.
        obj_mask = resize(crop(mask_obj, crop_center, crop_size), self.input_size) / 255.
        xywh = np.concatenate([crop_center - crop_size // 2, np.array([crop_size, crop_size])])
        Kroi = self.compute_K_roi(xywh, rgb_full.shape[1], rgb_full.shape[0])
        # mask bkg out
        mask_comb = (person_mask > 0.5) | (obj_mask > 0.5)
        rgb = rgb * np.expand_dims(mask_comb, -1)
        return Kroi, obj_mask, person_mask, rgb

    def upsample_predicted_pc(self, pc_obj):
        num_samples = int(self.num_samples*self.sample_ratio_hum) # here we assume same number of samples for human and object
        if len(pc_obj) > num_samples:
            # TODO: use farthest point sample
            ind_obj = np.random.choice(len(pc_obj), num_samples)
        else:
            # original pc + some random points
            ind_obj = np.concatenate([np.arange(len(pc_obj)), np.random.choice(len(pc_obj), num_samples - len(pc_obj))])
        pc_obj = pc_obj.copy()[ind_obj]
        return pc_obj


class BehaveCrossAttnTest(BehaveCrossAttnDataset):
    def get_samples(self, rgb_file):
        "return dummy samples"
        assert self.test_transl_type in ['estimated', 'estimated-2d'], f'invalid translation type {self.test_transl_type} for test only data'
        num_smpl = int(self.sample_ratio_hum * self.num_samples)
        num_obj = self.num_samples - num_smpl
        samples_smpl = np.random.randn(num_smpl, 3)
        samples_obj = np.random.randn(num_obj, 3)

        return samples_smpl, samples_obj
