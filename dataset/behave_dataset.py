"""
Dateset for BEHAVE, InterCap and ProciGen like datasets

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

from PIL import Image
from PIL.ImageFilter import GaussianBlur

from .behave_paths import DataPaths, date_seqs
from behave.kinect_transform import KinectTransform
from .img_utils import compute_translation, crop, masks2bbox, resize
from .utils import create_grid_points


class BehaveDataset(Dataset):
    def __init__(self, data_paths, num_samples, fix_sample=True,
                 input_size=(224, 224), split=None,
                 sample_ratio_hum=0.5,
                 normalize_type='comb',
                 smpl_type='gt',
                 uniform_obj_sample=False,
                 test_transl_type='norm',
                 bkg_type='none',
                 pred_binary=False,
                 ho_segm_pred_path=None, # path where segmented pc from stage model is saved
                 cam_noise_std=0., # camera pose noise standard deviation
                 sep_same_crop=False, # send same input image crop to separate models
                 aug_blur=0.0, # blurry augmentation
                 std_coverage=3.5,
                 debug=False,
                 behave_path='',
                 procigen_path=''
                 ):
        self.data_paths = data_paths
        self.num_samples = num_samples
        self.fix_sample = fix_sample
        self.sample_ratio_hum = sample_ratio_hum
        self.split = split
        self.input_size = input_size
        self.samples_cache = {}
        self.meshes_cache = {}

        self.kin_transforms = {
            f"Date{d:02d}": KinectTransform(date_seqs[f'Date{d:02d}'](behave_path), no_intrinsic=True) for d in range(1,8)
        }
        self.kin_transforms = {
            **self.kin_transforms,
            **{"Date09": KinectTransform(date_seqs['Date09'](procigen_path), no_intrinsic=True)}
        }

        # some constants
        # camera transform from opencv to pytorch3d
        opencv2py3d = np.eye(4)
        opencv2py3d[0, 0] = opencv2py3d[1, 1] = -1
        self.opencv2py3d = opencv2py3d

        # normalization type
        self.normalize_type = normalize_type # the way to normalize sampled points
        assert self.normalize_type in ['comb', 'smpl', 'bbox-comb', 'bbox-comb-scale', 'bbox-smpl', 'bbox-maxscale']
        self.smpl_type = smpl_type # use GT or other types
        assert self.smpl_type in ['gt', 'keypoints']
        if self.smpl_type == 'keypoints':
            # for evaluation/test
            assert self.normalize_type == 'smpl'
            print("Using keypoints to initialize camera parameters")

        # shapenet paths
        self.shapenet_corr_root = "/BS/xxie-2/static00/shapenet"
        self.uniform_obj_sample = uniform_obj_sample # combined with object

        # for direct translation predictor
        self.test_transl_type = test_transl_type
        assert self.test_transl_type in ['norm', 'estimated', 'normalized', 'estimated-2d'], f'unknown transl {test_transl_type}!'
        self.std_coverage = std_coverage # for estimating camera translation
        self.bkg_type = bkg_type # if none: mask bkg out, otherwise keep them
        print("std value used to estimate t:", std_coverage)
        assert self.bkg_type in ['none', 'original']

        self.focal = np.array([979.7844, 979.840]) # TODO: take InterCap parameters into account
        self.principal_point = np.array([1018.952, 779.486])
        self.behave_image_size = np.array([2048, 1536])
        self.icap_image_size = np.array([1920, 1080])

        # binary segmentation
        self.predict_binary = pred_binary
        self.grid_points, self.reso = None, 96
        if self.predict_binary:
            # generate grid samples
            self.grid_points = create_grid_points(0.5, self.reso) # 1M points

        self.ho_segm_pred_path = ho_segm_pred_path
        assert self.smpl_type == 'gt', f'must use GT SMPL mesh instead of {self.smpl_type}'
        # assert self.test_transl_type == 'estimated-2d', f'only support estimated-2d estimation instead of {self.test_transl_type}'
        self.cam_noise_std = cam_noise_std
        self.sep_same_crop = sep_same_crop
        print("Use the same crop as input to separate models?", sep_same_crop)

        self.aug_blur = aug_blur # random blurness augmentation
        # print(f"Aug_blur value={aug_blur}...")

        self.DEBUG = debug

    def __getitem__(self, idx):
        # ret = self.get_item(idx)
        # return ret
        try:
            ret = self.get_item(idx)
            return ret
        except Exception as e:
            print(e)
            ridx = np.random.randint(0, len(self.data_paths))
            print(f"failed on {self.data_paths[idx]}, retrying {self.data_paths[ridx]}")
            return self[ridx]

    def __len__(self):
        return len(self.data_paths)

    def blur_image(self, img, aug_blur):
        assert isinstance(img, np.ndarray)
        if aug_blur > 0.000001:
            # the Gaussian radius, i.e. std of the 2d gaussian function
            x = np.random.uniform(0, aug_blur) * 224.  # input image is in range [0, 255]
            blur = GaussianBlur(x)
            img = Image.fromarray(img)
            # print(f"Blurring image with std={x:.4f}")
            return np.array(img.filter(blur))
        return img

    def get_item(self, idx):
        """

        Parameters
        ----------
        idx

        Returns
        -------
            pclouds
            images
            masks (2, h, w)
            R, T, K for camera projection
            sequence_name
            synset_id


        """
        # start = time.time()
        rgb_file = self.data_paths[idx]
        mask_hum, mask_obj = self.load_masks(rgb_file)
        rgb_full = cv2.imread(rgb_file)[:, :, ::-1]
        color_h, color_w = rgb_full.shape[:2]

        if self.split == 'train' and np.random.uniform()>0.5:
            rgb_full = self.blur_image(rgb_full, self.aug_blur)

        # crop
        bmax, bmin, crop_center, crop_size = self.get_crop_params(mask_hum, mask_obj)
        rgb = resize(crop(rgb_full, crop_center, crop_size), self.input_size) / 255.
        person_mask = resize(crop(mask_hum, crop_center, crop_size), self.input_size) / 255.
        obj_mask = resize(crop(mask_obj, crop_center, crop_size), self.input_size) / 255.

        # mask bkg out
        mask_comb = (person_mask > 0.5) | (obj_mask > 0.5)
        rgb = rgb * np.expand_dims(mask_comb, -1)

        samples_obj, samples_smpl = self.get_samples(rgb_file)
        samples = np.concatenate([samples_smpl, samples_obj], 0)

        # now compute the camera parameters
        cent, radius = self.get_normalize_params(samples, samples_smpl)
        samples = samples - cent # when using estimated translation, it should be another scale value that projects GT to image
        samples = samples / (radius * 2) # now it fits into a unit cube

        # estimate translation: this always uses full crop
        bmin_ho, bmax_ho = masks2bbox([mask_hum, mask_obj])
        crop_size_ho = int(np.max(bmax_ho - bmin_ho) * 1.0)
        if crop_size_ho % 2 == 1:
            crop_size_ho += 1  # make sure it is an even number
        is_behave = self.is_behave_dataset(rgb_full.shape[1])

        cent_transform = np.eye(4)  # the transform applied to the mesh that moves it back to kinect camera frame
        transl_estimate = np.zeros(4) # dummy data
        if self.test_transl_type == 'estimated-2d':
            assert rgb_full.shape[1] in [2048, 1920], 'the image is not normalized to BEHAVE or ICAP size!'
            indices = np.indices(rgb_full.shape[:2])
            assert np.sum(mask_obj > 127) > 5, f'not enough object mask found for {rgb_file}'
            pts_h = np.stack([indices[1][mask_hum > 127], indices[0][mask_hum > 127]], -1)
            pts_o = np.stack([indices[1][mask_obj > 127], indices[0][mask_obj > 127]], -1)
            proj_cent_est = (np.mean(pts_h, 0) + np.mean(pts_o, 0)) / 2.
            transl_estimate = compute_translation(proj_cent_est, crop_size_ho, is_behave, self.std_coverage)
            cent_transform[:3, 3] = transl_estimate / 7.0
            radius = 0.5  # don't do normalization anymore
            cent = transl_estimate / 7.0
            # print(f"Estimated 2d: {proj_cent_est}, 3D: {transl_estimate/7.}")

        comb = np.matmul(self.opencv2py3d, cent_transform)
        R = torch.from_numpy(comb[:3, :3]).float()
        T = torch.from_numpy(comb[:3, 3]).float() / (radius * 2)

        xywh = np.concatenate([crop_center - crop_size//2, np.array([crop_size, crop_size])])
        Kroi = self.compute_K_roi(xywh, rgb_full.shape[1], rgb_full.shape[0])

        # compute scale and translation for human, object separately
        num_smpl = int(self.sample_ratio_hum * self.num_samples)

        ss = rgb_file.split(os.sep)
        data_dict = {
            "R": R,
            "T": T,
            "K": torch.from_numpy(Kroi).float(),
            "image_path":rgb_file,
            'view_id': DataPaths.get_kinect_id(rgb_file),
            "image_size_hw": torch.tensor(self.input_size),
            "images":torch.from_numpy(rgb).float().permute(2, 0, 1),
            "masks":torch.from_numpy(np.stack([person_mask, obj_mask], 0)).float(),
            "sequence_name": ss[-2],# the frame name
            "synset_id":ss[-3],
            "pclouds":torch.from_numpy(samples).float(),
            'orig_image_size': torch.tensor([color_h, color_w]),

            # some normalization parameters
            "gt_trans":cent,
            'radius': radius,
            "estimated_trans": transl_estimate,
            "crop_center":crop_center,
            "crop_size":crop_size,
            "num_smpl": num_smpl
        }

        if self.predict_binary and self.split=='train':
            assert self.fix_sample
            img_key = self.get_cache_key(rgb_file)
            smpl, obj, grid_df = self.meshes_cache[img_key]
            if grid_df is None:
                # Load the precomputed human/object occupancy
                res = 128
                npz_file = rgb_file.replace(".color.jpg", f".grid_df_res{res}_b0.5.npz")
                occupancies = np.unpackbits(np.load(npz_file)['compressed_occ'])
                grid_df = np.reshape(occupancies, (res,) * 3).astype(float)  # human-1, obj-0
                self.meshes_cache[img_key] = None, None, grid_df

            data_dict['grid_df'] = torch.from_numpy(grid_df).float()
            # for debug
            if self.DEBUG:
                data_dict['verts_obj'] = (np.array(obj.vertices) - cent) / (2 * radius)
                data_dict['verts_smpl'] = (np.array(smpl.vertices) - cent)/(2*radius)
                data_dict['faces_obj'] = np.array(obj.faces)
                data_dict['faces_smpl'] = np.array(smpl.faces)

        if self.ho_segm_pred_path is not None:
            # load predicted h+o
            pred_dict = self.load_predicted_HO(cent, radius, rgb_file)

            data_dict = {
                **data_dict,
                **pred_dict
            }
        data_dict = self.add_additional_data(data_dict)
        return data_dict

    def add_additional_data(self, data_dict):
        return data_dict

    def load_predicted_HO(self, cent, radius, rgb_file):
        # print("Loading predicted HO from", self.ho_segm_pred_path)
        ss = rgb_file.split(os.sep)
        pred_file = osp.join(self.ho_segm_pred_path, ss[-3], ss[-2] + ".ply")
        if not osp.isfile(pred_file):
            pred_file = osp.join(self.ho_segm_pred_path, ss[-3], ss[-2] + ".pkl")
            assert osp.isfile(pred_file), f'{pred_file} does noto exist!'
            print(f"Warning: loading direct trans+scale prediction from {pred_file}")

            # PC and object are dummy points
            num_samples = int(self.num_samples * self.sample_ratio_hum)
            pc_hum, pc_obj = np.zeros((num_samples, 3)), np.zeros((num_samples, 3))

            # scale and t are predicted directly
            pred_params = pkl.load(open(pred_file, 'rb'))
            pred_trans = pred_params['pred_trans'] # (6, ) H +O
            pred_scale = pred_params['pred_scale'] # (2, ) H+O
            cent_hum, cent_obj = pred_trans[:3], pred_trans[3:]
            scale_hum, scale_obj = pred_scale[0], pred_scale[1]
        else:
            pc = trimesh.load_mesh(pred_file, process=False)
            mask_hum = pc.colors[:, 2] > 0.5
            # print(f'{np.sum(mask_hum)}/{len(mask_hum)} human points')
            pc_hum, pc_obj = np.array(pc.vertices[mask_hum]), np.array(pc.vertices[~mask_hum])
            # compute t of human and object and then do normalization
            cent_hum, cent_obj = np.mean(pc_hum, 0), np.mean(pc_obj, 0)
            scale_hum = np.sqrt(np.max(np.sum((pc_hum - cent_hum) ** 2, -1)))
            scale_obj = np.sqrt(np.max(np.sum((pc_obj - cent_obj) ** 2, -1)))
        T_ho_scaled = cent / (radius * 2) # compute translation for H+O space

        # apply to camera parameter, and points
        pc_hum = (pc_hum - cent_hum) / (2 * scale_hum)
        pc_obj = (pc_obj - cent_obj) / (2 * scale_obj)
        T_hum_scaled = (T_ho_scaled + cent_hum) / (2 * scale_hum)
        T_obj_scaled = (T_ho_scaled + cent_obj) / (2 * scale_obj)
        # apply opencv to pytorch3d transform: flip x and y
        T_hum_scaled *= np.array([-1, -1, 1])
        T_obj_scaled *= np.array([-1, -1, 1])
        # up-sample points to have same shape as H+O
        pc_hum = self.upsample_predicted_pc(pc_hum)
        pc_obj = self.upsample_predicted_pc(pc_obj)
        # print("number of points after up-sampling:", pc_obj.shape, pc_hum.shape, self.num_samples)

        pred_dict = {
            "pred_hum":  torch.from_numpy(pc_hum).float(),
            "pred_obj": torch.from_numpy(pc_obj).float(),
            "T_obj_scaled": torch.from_numpy(T_obj_scaled).float(),
            "T_hum_scaled": torch.from_numpy(T_hum_scaled).float(),

            # also normalization parameters
            "cent_hum_pred": torch.from_numpy(cent_hum).float(),
            "cent_obj_pred": torch.from_numpy(cent_obj).float(),
            "radius_hum_pred": torch.tensor([scale_hum]).float(),
            "radius_obj_pred": torch.tensor([scale_obj]).float(),
        }

        return pred_dict

    def upsample_predicted_pc(self, pc_obj):
        ind_obj = np.concatenate([np.arange(len(pc_obj)), np.random.choice(len(pc_obj), self.num_samples - len(pc_obj))])
        pc_obj = pc_obj.copy()[ind_obj]
        return pc_obj

    def get_normalize_params(self, samples, samples_smpl):
        """
        computer center and scale for the given normalization type
        :param samples: (N, 3)
        :param samples_smpl: (N_h, 3)
        :return:
        """
        if self.normalize_type == 'smpl':
            # use points sampled from SMPL mesh surface to do normalization
            cent, radius = self.unit_sphere_params(samples_smpl)
        elif self.normalize_type == 'bbox-smpl':
            bmin, bmax = np.min(samples_smpl, 0), np.max(samples_smpl, 0)
            cent = (bmax + bmin) / 2
            radius = np.sqrt(np.sum((bmax - bmin) ** 2)) / 2.0
        elif self.normalize_type == 'bbox-comb':
            # use bbox center instead of mean center
            bmin, bmax = np.min(samples, 0), np.max(samples, 0)
            cent = (bmax + bmin) / 2
            radius = np.sqrt(np.max(np.sum((samples - cent) ** 2, -1)))
        elif self.normalize_type == 'bbox-comb-scale':
            # use bbox center, and bbox diameter as scale factor
            bmin, bmax = np.min(samples, 0), np.max(samples, 0)
            cent = (bmax + bmin) / 2
            radius = np.sqrt(np.sum((bmax - bmin) ** 2)) / 2.0
        elif self.normalize_type == 'bbox-maxscale':
            bmin, bmax = np.min(samples, 0), np.max(samples, 0)
            cent = (bmax + bmin) / 2
            radius = 1.25  # use max scale, make sure all objects are fixed inside this bbox
        else:
            # simply take the center of gravity
            cent, radius = self.unit_sphere_params(samples)
        return cent, radius

    def unit_sphere_params(self, samples):
        cent = np.mean(samples, 0)
        radius = np.sqrt(np.max(np.sum((samples - cent) ** 2, -1)))
        return cent, radius

    def get_crop_params(self, mask_hum, mask_obj, bbox_exp=1.0):
        "compute bounding box based on masks"
        bmin, bmax = masks2bbox([mask_hum, mask_obj])
        crop_center = (bmin + bmax) // 2
        crop_size = int(np.max(bmax - bmin) * bbox_exp)
        if crop_size % 2 == 1:
            crop_size += 1  # make sure it is an even number
        return bmax, bmin, crop_center, crop_size

    def get_samples(self, rgb_file):
        """
        sample SMPL and object surface
        Parameters
        ----------
        rgb_file

        Returns
        -------

        """
        img_key = self.get_cache_key(rgb_file)
        if self.fix_sample and img_key in self.samples_cache:
            # do not do re-sample, use the last samples
            samples_smpl, samples_obj = self.samples_cache[img_key]
        else:
            # do sample
            smpl_name, obj_name = self.get_gt_fit_names(rgb_file)
            smpl_path = self.get_smpl_filepath(rgb_file, smpl_name, obj_name)
            obj_path = self.get_obj_filepath(rgb_file, smpl_name, obj_name)
            smpl = trimesh.load_mesh(smpl_path, process=False)
            obj = self.load_obj_gtmesh(obj_path)
            date, kid = DataPaths.get_seq_date(rgb_file), DataPaths.get_kinect_id(rgb_file)
            # transform to local
            smpl.vertices = self.kin_transforms[date].world2local(smpl.vertices, kid)
            obj.vertices = self.kin_transforms[date].world2local(obj.vertices, kid)
            num_smpl = int(self.sample_ratio_hum * self.num_samples)
            num_obj = self.num_samples - num_smpl
            samples_smpl = smpl.sample(num_smpl)
            samples_obj = obj.sample(num_obj)

            # Save samples to cache for fast access next time
            self.samples_cache[img_key] = (samples_smpl, samples_obj)
        return samples_obj, samples_smpl

    def get_cache_key(self, rgb_file):
        ss = rgb_file.split(os.sep)
        img_key = str(os.sep).join(ss[-3:])
        return img_key

    def get_obj_filepath(self, rgb_file, smpl_name, obj_name):
        obj_path = DataPaths.rgb2obj_path(rgb_file, obj_name)
        return obj_path

    def get_smpl_filepath(self, rgb_file, smpl_name, obj_name):
        smpl_path = DataPaths.rgb2smpl_path(rgb_file, smpl_name)
        return smpl_path

    def compute_K_roi(self, bbox_square,
                      image_width=2048,
                      image_height=1536):
        """
        compute Kroi for the cropped patch, return results in ndc coordinate
        :param bbox_square: (x,y,b,w) of the patch
        :param image_width: width of the original image
        :param image_height: height of the original image
        :return:
        """
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

    def is_behave_dataset(self, image_width):
        assert image_width in [2048, 1920, 1024, 960], f'unknwon image width {image_width}!'
        if image_width in [2048, 1024]:
            is_behave = True
        else:
            is_behave = False
        return is_behave

    def load_masks(self, rgb_file, flip=False):
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

    def load_obj_gtmesh(self, obj_file):
        return trimesh.load_mesh(obj_file, process=False)

    def get_gt_fit_names(self, rgb_file):
        "based on the rgb file path, find the smpl and object fit name"
        seq_name = DataPaths.get_seq_name(rgb_file)
        if 'Date04_Subxx' in seq_name or 'Date09_Subxx' in seq_name:
            # synthetic ProciGen dataset
            return 'fit01', 'fit01'
        elif "Date0" in seq_name:
            # behave real sequences 
            # TODO: if you are using behvae-30fps data, change this to 'fit03', 'fit01-smooth'
            return 'fit02', 'fit01'
        else:
            # for InterCap or other dataset
            return 'fit02', 'fit01'


class BehaveTestOnly(BehaveDataset):
    "for test only, do not load any 3D meshes or samples, only return random points"
    def get_samples(self, rgb_file):
        """not loading real 3D points"""
        assert self.test_transl_type in ['estimated', 'estimated-2d'], f'invalid translation type {self.test_transl_type} for test only data'
        num_smpl = int(self.sample_ratio_hum * self.num_samples)
        num_obj = self.num_samples - num_smpl
        samples_smpl = np.random.randn(num_smpl, 3)
        samples_obj = np.random.randn(num_obj, 3)

        return samples_smpl, samples_obj