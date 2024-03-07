import os
import numpy as np
import cv2
import torch

from .base_data import BaseDataset
from .behave_paths import DataPaths
from .img_utils import compute_translation, masks2bbox, crop


def padTo_4x3(rgb, person_mask, obj_mask, aspect_ratio=0.75):
    """
    pad images to have 4:3 aspect ratio
    :param rgb: (H, W, 3)
    :param person_mask:
    :param obj_mask:
    :return: all images at the given aspect ratio
    """
    h, w = rgb.shape[:2]
    if w > h * 1/aspect_ratio:
        # pad top
        h_4x3 = int(w * aspect_ratio)
        pad_top = h_4x3 - h
        rgb_pad = np.pad(rgb, ((pad_top, 0), (0, 0), (0, 0)))
        person_mask = np.pad(person_mask, ((pad_top, 0), (0, 0))) if person_mask is not None else None
        obj_mask = np.pad(obj_mask, ((pad_top, 0), (0, 0))) if obj_mask is not None else None
    else:
        # pad two side
        w_new = np.lcm.reduce([h * 2, 16]) # least common multiplier
        h_4x3 = int(w_new * aspect_ratio)
        pad_top = h_4x3 - h
        pad_left = (w_new - w) // 2
        pad_right = w_new - w - pad_left
        rgb_pad = np.pad(rgb, ((pad_top, 0), (pad_left, pad_right), (0, 0)))
        obj_mask = np.pad(obj_mask, ((pad_top, 0), (pad_left, pad_right))) if obj_mask is not None else None
        person_mask = np.pad(person_mask, ((pad_top, 0), (pad_left, pad_right))) if person_mask is not None else None
    return rgb_pad, obj_mask, person_mask


def recrop_input(rgb, person_mask, obj_mask, dataset_name='behave'):
    "recrop input images"
    exp_ratio = 1.42
    if dataset_name == 'behave':
        mean_center = np.array([1008, 995])  # mean RGB image crop center
        behave_size = (2048, 1536)
        new_size = (int(750 * exp_ratio), int(exp_ratio * 750))
    else:
        mean_center = np.array([904, 668])  # mean RGB image crop center for bottle sequences of ICAP
        behave_size = (1920, 1080)
        new_size = (int(593.925 * exp_ratio), int(exp_ratio * 593.925))  # mean width of bottle sequences
    aspect_ratio = behave_size[1] / behave_size[0]
    pad_top = mean_center[1] - new_size[0] // 2
    pad_bottom = behave_size[1] - (mean_center[1] + new_size[0] // 2)
    pad_left = mean_center[0] - new_size[0] // 2
    pad_right = behave_size[0] - (mean_center[0] + new_size[0] // 2)

    # First resize to the same aspect ratio
    if rgb.shape[0] / rgb.shape[1] != aspect_ratio:
        rgb, obj_mask, person_mask = padTo_4x3(rgb, person_mask, obj_mask, aspect_ratio)

    # Resize to the same size as behave image, to have a comparable pixel size
    rgb = cv2.resize(rgb, behave_size)
    mask_ps = cv2.resize(person_mask, behave_size)
    mask_obj = cv2.resize(obj_mask, behave_size)

    # Crop and resize the human + object patch
    bmin, bmax = masks2bbox([mask_ps, mask_obj])
    center = (bmin + bmax) // 2
    crop_size = int(np.max(bmax - bmin) * exp_ratio)  # larger crop to have background
    img_crop = cv2.resize(crop(rgb, center, crop_size), new_size)
    mask_ps = cv2.resize(crop(mask_ps, center, crop_size), new_size)
    mask_obj = cv2.resize(crop(mask_obj, center, crop_size), new_size)

    # Pad back to have same shape as behave image
    img_full = np.pad(img_crop, [[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    mask_ps_full = np.pad(mask_ps, [[pad_top, pad_bottom], [pad_left, pad_right]])
    mask_obj_full = np.pad(mask_obj, [[pad_top, pad_bottom], [pad_left, pad_right]])

    # Make sure the image shape is the same
    if img_full.shape[:2] != behave_size[::-1]:
        img_full = cv2.resize(img_full, behave_size)
        mask_ps_full = cv2.resize(mask_ps_full, behave_size)
        mask_obj_full = cv2.resize(mask_obj_full, behave_size)
    return img_full, mask_ps_full, mask_obj_full


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

        return self.image2dict(mask_hum, mask_obj, rgb_full, rgb_file)

    def image2dict(self, mask_hum, mask_obj, rgb_full, rgb_file=None):
        "do all the necessary preprocessing for images"
        if rgb_full.shape[:2] != mask_obj.shape[:2]:
            raise ValueError(f"The given object mask shape {mask_obj.shape[:2]} does not match the RGB image shape {rgb_full.shape[:2]}")
        if rgb_full.shape[:2] != mask_hum.shape[:2]:
            raise ValueError(f"The given human mask shape {mask_hum.shape[:2]} does not match the RGB image shape {rgb_full.shape[:2]}")

        if rgb_full.shape[:2] not in [(1080, 1920), (1536, 2048)]:
            # crop and resize the image to behave image size
            print(f"Recropping the input image and masks for {rgb_file}")
            rgb_full, mask_hum, mask_obj = recrop_input(rgb_full, mask_hum, mask_obj)
        color_h, color_w = rgb_full.shape[:2]
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
        if rgb_full.shape[1] not in [2048, 1920]:
            raise ValueError('the image is not normalized to BEHAVE or ICAP size!')
        indices = np.indices(rgb_full.shape[:2])
        if np.sum(mask_obj > 127) < 5:
            raise ValueError(f'not enough object mask found for {rgb_file}')
        pts_h = np.stack([indices[1][mask_hum > 127], indices[0][mask_hum > 127]], -1)
        pts_o = np.stack([indices[1][mask_obj > 127], indices[0][mask_obj > 127]], -1)
        proj_cent_est = (np.mean(pts_h, 0) + np.mean(pts_o, 0)) / 2.  # heuristic to obtain 2d projection center
        transl_estimate = compute_translation(proj_cent_est, crop_size_ho, is_behave, self.std_coverage)
        cent_transform[:3, 3] = transl_estimate / 7.0
        radius = 0.5  # don't do normalization anymore
        cent = transl_estimate / 7.0
        comb = np.matmul(self.opencv2py3d, cent_transform)
        R = torch.from_numpy(comb[:3, :3]).float()
        T = torch.from_numpy(comb[:3, 3]).float() / (radius * 2)
        data_dict = {
            "R": R,
            "T": T,
            "K": torch.from_numpy(Kroi).float(),
            "T_ho": torch.from_numpy(cent).float(),  # translation for H+O
            "image_path": rgb_file,
            "image_size_hw": torch.tensor(self.input_size),
            "images": torch.from_numpy(rgb_fullcrop).float().permute(2, 0, 1),
            "masks": torch.from_numpy(np.stack([psmask_fullcrop, objmask_fullcrop], 0)).float(),
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

    def image2batch(self, rgb, mask_hum, mask_obj):
        """
        given input image, convert it into a batch object ready for model inference
        :param rgb: (h, w, 3), np array
        :param mask_hum: (h, w, 3), np array
        :param mask_obj: (h, w, 3), np array
        :return:
        """
        mask_hum = np.mean(mask_hum, -1)
        mask_obj = np.mean(mask_obj, -1)

        data_dict = self.image2dict(mask_hum, mask_obj, rgb, 'input image')
        # convert dict to list
        new_dict = {k:[v] for k, v in data_dict.items()}

        return new_dict


