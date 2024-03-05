from os import path as osp

import cv2
import numpy as np
from torch.utils.data import Dataset

from dataset.img_utils import masks2bbox, resize, crop


class BaseDataset(Dataset):
    def __init__(self, data_paths, input_size=(224, 224)):
        self.data_paths = data_paths # RGB image files
        self.input_size = input_size
        opencv2py3d = np.eye(4)
        opencv2py3d[0, 0] = opencv2py3d[1, 1] = -1
        self.opencv2py3d = opencv2py3d

    def __len__(self):
        return len(self.data_paths)

    def load_masks(self, rgb_file):
        person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.png")
        if not osp.isfile(person_mask_file):
            person_mask_file = rgb_file.replace('.color.jpg', ".person_mask.jpg")
        obj_mask_file = None
        for pat in [".obj_rend_mask.png", ".obj_rend_mask.jpg", ".obj_mask.png", ".obj_mask.jpg", ".object_rend.png"]:
            obj_mask_file = rgb_file.replace('.color.jpg', pat)
            if osp.isfile(obj_mask_file):
                break
        person_mask = cv2.imread(person_mask_file, cv2.IMREAD_GRAYSCALE)
        obj_mask = cv2.imread(obj_mask_file, cv2.IMREAD_GRAYSCALE)

        return person_mask, obj_mask

    def get_crop_params(self, mask_hum, mask_obj, bbox_exp=1.0):
        "compute bounding box based on masks"
        bmin, bmax = masks2bbox([mask_hum, mask_obj])
        crop_center = (bmin + bmax) // 2
        # crop_size = np.max(bmax - bmin)
        crop_size = int(np.max(bmax - bmin) * bbox_exp)
        if crop_size % 2 == 1:
            crop_size += 1  # make sure it is an even number
        return bmax, bmin, crop_center, crop_size

    def is_behave_dataset(self, image_width):
        assert image_width in [2048, 1920, 1024, 960], f'unknwon image width {image_width}!'
        if image_width in [2048, 1024]:
            is_behave = True
        else:
            is_behave = False
        return is_behave

    def compute_K_roi(self, bbox_square,
                      image_width=2048,
                      image_height=1536,
                      fx=979.7844, fy=979.840,
                      cx=1018.952, cy=779.486):
        "return results in ndc coordinate, this is correct!!!"
        x, y, b, w = bbox_square
        assert b == w
        is_behave = self.is_behave_dataset(image_width)

        if is_behave:
            assert image_height / image_width == 0.75, f"invalid image aspect ratio: width={image_width}, height={image_height}"
            # the image might be rendered at different size
            ratio = image_width/2048.
            fx, fy = 979.7844*ratio, 979.840*ratio
            cx, cy = 1018.952*ratio, 779.486*ratio
        else:
            assert image_height / image_width == 9/16, f"invalid image aspect ratio: width={image_width}, height={image_height}"
            # intercap camera
            ratio = image_width/1920
            fx, fy = 918.457763671875*ratio, 918.4373779296875*ratio
            cx, cy = 956.9661865234375*ratio, 555.944580078125*ratio

        cx, cy = cx - x, cy - y
        scale = b/2.
        # in ndc
        cx_ = (scale - cx)/scale
        cy_ = (scale - cy)/scale
        fx_ = fx/scale
        fy_ = fy/scale

        K_roi = np.array([
            [fx_, 0, cx_, 0],
            [0., fy_, cy_, 0, ],
            [0, 0, 0, 1.],
            [0, 0, 1, 0]
        ])
        return K_roi

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
