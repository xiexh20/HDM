import os
import numpy as np
import cv2
import torch

from .base_data import BaseDataset
from .behave_paths import DataPaths
from .img_utils import compute_translation, masks2bbox


class DemoDataset(BaseDataset):
    def __init__(self, data_paths, input_size=(224, 224),
                 std_coverage=3.5, # used to estimate camera translation
                 ):
        super().__init__(data_paths, input_size)
        self.std_coverage = std_coverage

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        rgb_file = self.data_paths[idx]
        mask_hum, mask_obj = self.load_masks(rgb_file)

        rgb_full = cv2.imread(rgb_file)[:, :, ::-1]
        color_h, color_w = rgb_full.shape[:2]
        # TODO: preprocess image in case the RGB image size is not the same as BEHAVE image size

        # Input to the first stage model: human + object crop
        Kroi, objmask_fullcrop, psmask_fullcrop, rgb_fullcrop = self.crop_full_image(mask_hum.copy(),
                                                                                     mask_obj.copy(),
                                                                                     rgb_full.copy(),
                                                                                     [mask_hum, mask_obj],
                                                                                     1.00)

        # Input to the second stage model: human and object crops
        Kroi_h, masko_hum, maskh_hum, rgb_hum = self.crop_full_image(mask_hum.copy(),
                                                                  mask_obj.copy(),
                                                                  rgb_full.copy(),
                                                                  [mask_hum, mask_hum], 1.05)
        Kroi_o, masko_obj, maskh_obj, rgb_obj = self.crop_full_image(mask_hum.copy(),
                                                                  mask_obj.copy(),
                                                                  rgb_full.copy(),
                                                                  [mask_obj, mask_obj], 1.5)

        # Estimate camera translation
        cent_transform = np.eye(4)  # the transform applied to the mesh that moves it back to kinect camera frame
        bmin_ho, bmax_ho = masks2bbox([mask_hum, mask_obj])
        crop_size_ho = int(np.max(bmax_ho - bmin_ho) * 1.0)
        if crop_size_ho % 2 == 1:
            crop_size_ho += 1  # make sure it is an even number
        is_behave = self.is_behave_dataset(rgb_full.shape[1])
        assert rgb_full.shape[1] in [2048, 1920], 'the image is not normalized to BEHAVE or ICAP size!'
        indices = np.indices(rgb_full.shape[:2])
        assert np.sum(mask_obj > 127) > 5, f'not enough object mask found for {rgb_file}'
        pts_h = np.stack([indices[1][mask_hum > 127], indices[0][mask_hum > 127]], -1)
        pts_o = np.stack([indices[1][mask_obj > 127], indices[0][mask_obj > 127]], -1)
        proj_cent_est = (np.mean(pts_h, 0) + np.mean(pts_o, 0)) / 2. # heuristic to obtain 2d projection center
        transl_estimate = compute_translation(proj_cent_est, crop_size_ho, is_behave, self.std_coverage)
        cent_transform[:3, 3] = transl_estimate / 7.0
        radius = 0.5  # don't do normalization anymore
        cent = transl_estimate / 7.0

        comb = np.matmul(self.opencv2py3d, cent_transform)
        R = torch.from_numpy(comb[:3, :3]).float()
        T = torch.from_numpy(comb[:3, 3]).float() / (radius * 2)

        ss = rgb_file.split(os.sep)
        data_dict = {
            "R": R,
            "T": T,
            "K": torch.from_numpy(Kroi).float(),
            "T_ho": torch.from_numpy(cent).float(), # translation for H+O
            "image_path": rgb_file,
            'view_id': DataPaths.get_kinect_id(rgb_file),
            "image_size_hw": torch.tensor(self.input_size),
            "images": torch.from_numpy(rgb_fullcrop).float().permute(2, 0, 1),
            "masks": torch.from_numpy(np.stack([psmask_fullcrop, objmask_fullcrop], 0)).float(),
            "sequence_name": ss[-2],  # the frame name
            "synset_id": ss[-3],
            'orig_image_size': torch.tensor([color_h, color_w]),

            # Human input to stage 2
            "images_hum": torch.from_numpy(rgb_hum).float().permute(2, 0, 1),
            "masks_hum": torch.from_numpy(np.stack([maskh_hum, masko_hum], 0)).float(),
            "K_hum": torch.from_numpy(Kroi_h).float(),

            # Object input to stage 2
            "images_obj": torch.from_numpy(rgb_obj).float().permute(2, 0, 1),
            "masks_obj": torch.from_numpy(np.stack([maskh_obj, masko_obj], 0)).float(),
            "K_obj": torch.from_numpy(Kroi_o).float(),

            # some normalization parameters
            "gt_trans": cent,
            'radius': radius,
            "estimated_trans": transl_estimate,
        }

        return data_dict


