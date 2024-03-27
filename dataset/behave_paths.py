import glob
import os, re
import pickle as pkl
from os.path import join, basename, dirname, isfile
import os.path as osp

import cv2, json
import numpy as np


class DataPaths:
    """
    class to handle path operations based on BEHAVE dataset structure
    """
    def __init__(self):
        pass

    @staticmethod
    def load_splits(split_file, dataset_path=None):
        assert os.path.exists(dataset_path), f'the given dataset path {dataset_path} does not exist, please check if your training data are placed over there!'
        train, val = DataPaths.get_train_test_from_pkl(split_file)
        if isinstance(train[0], list):
            # video data
            train_full = [[join(dataset_path, seq[x]) for x in range(len(seq))] for seq in train]
            val_full = [[join(dataset_path, seq[x]) for x in range(len(seq))] for seq in val]
        else:
            train_full = [join(dataset_path, x) for x in train] # full path to the training data
            val_full = [join(dataset_path, x) for x in val] # full path to the validation data files

        return train_full, val_full

    @staticmethod
    def load_splits_abs(split_file):
        "the file path stored in split_file is abs path"
        train, val = DataPaths.get_train_test_from_pkl(split_file)
        return train, val

    @staticmethod
    def get_train_test_from_pkl(pkl_file):
        data = pkl.load(open(pkl_file, 'rb'))
        return data['train'], data['test']

    @staticmethod
    def get_image_paths_seq(seq, tid=1, check_occlusion=False, pat='t*.000'):
        """
        find all image paths in one sequence
        :param seq: path to one behave sequence
        :param tid: test on images from which camera
        :param check_occlusion: whether to load full object mask and check occlusion ratio
        :return: a list of paths to test image files
        """
        image_files = sorted(glob.glob(seq + f"/{pat}/k{tid}.color.jpg"))
        # print(image_files, seq + f"/{pat}/k{tid}.color.jpg")
        if not check_occlusion:
            return image_files
        # check object occlusion ratio
        valid_files = []
        count = 0
        for img_file in image_files:
            mask_file = img_file.replace('.color.jpg', '.obj_rend_mask.png')
            if not os.path.isfile(mask_file):
                mask_file = img_file.replace('.color.jpg', '.obj_rend_mask.jpg')
            full_mask_file = img_file.replace('.color.jpg', '.obj_rend_full.png')
            if not os.path.isfile(full_mask_file):
                full_mask_file = img_file.replace('.color.jpg', '.obj_rend_full.jpg')
            if not isfile(mask_file) or not isfile(full_mask_file):
                continue

            mask = np.sum(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE) > 127)
            mask_full = np.sum(cv2.imread(full_mask_file, cv2.IMREAD_GRAYSCALE) > 127)
            if mask_full == 0:
                count += 1
                continue

            ratio = mask / mask_full
            if ratio > 0.3:
                valid_files.append(img_file)
            else:
                count += 1
                print(f'{mask_file} occluded by {1 - ratio}!')
        return valid_files

    @staticmethod
    def get_kinect_id(rgb_file):
        "extract kinect id from the rgb file"
        filename = osp.basename(rgb_file)
        try:
            kid = int(filename.split('.')[0][1])
            assert kid in [0, 1, 2, 3, 4, 5], f'found invalid kinect id {kid} for file {rgb_file}'
            return kid
        except Exception as e:
            print(rgb_file)
            raise ValueError()

    @staticmethod
    def get_seq_date(rgb_file):
        "date for the sequence"
        seq_name = str(rgb_file).split(os.sep)[-3]
        date = seq_name.split('_')[0]
        assert date in ['Date01', 'Date02', 'Date03', 'Date04', 'Date05', 'Date06', 'Date07',
                        "ICapS01", "ICapS02", "ICapS03", "Date08", "Date09"], f"invalid date for {rgb_file}"
        return date

    @staticmethod
    def rgb2obj_path(rgb_file:str, save_name='fit01-smooth'):
        "convert an rgb file to a obj mesh file"
        ss = rgb_file.split(os.sep)
        seq_name = ss[-3]
        obj_name = seq_name.split('_')[2]
        real_name = obj_name
        if 'chair' in obj_name:
            real_name = 'chair'
        if 'ball' in obj_name:
            real_name = 'sports ball'

        frame_folder = osp.dirname(rgb_file)
        mesh_file = osp.join(frame_folder, real_name, save_name, f'{real_name}_fit.ply')

        if not osp.isfile(mesh_file):
            # synthetic data
            mesh_file = osp.join(frame_folder, obj_name, save_name, f'{obj_name}_fit.ply')
        return mesh_file

    @staticmethod
    def rgb2smpl_path(rgb_file:str, save_name='fit03'):
        frame_folder = osp.dirname(rgb_file)
        real_name = 'person'
        mesh_file = osp.join(frame_folder, real_name, save_name, f'{real_name}_fit.ply')
        return mesh_file

    @staticmethod
    def rgb2seq_frame(rgb_file:str):
        "rgb file to seq_name, frame time"
        ss = rgb_file.split(os.sep)
        return ss[-3], ss[-2]

    @staticmethod
    def rgb2recon_folder(rgb_file, save_name, recon_path):
        "convert rgb file to the subfolder"
        dataset_path = osp.dirname(osp.dirname(osp.dirname(rgb_file)))
        recon_folder = osp.join(osp.dirname(rgb_file.replace(dataset_path, recon_path)), save_name)
        return recon_folder

    @staticmethod
    def get_seq_name(rgb_file):
        return osp.basename(osp.dirname(osp.dirname(rgb_file)))


    @staticmethod
    def rgb2object_name(rgb_file):
        seq_name = DataPaths.get_seq_name(rgb_file)
        obj_name = seq_name.split('_')[2]
        return obj_name

    @staticmethod
    def rgb2gender(rgb_file):
        "find the gender of this image"
        seq_name = str(rgb_file).split(os.sep)[-3]
        sub = seq_name.split('_')[1]
        return _sub_gender[sub]

    @staticmethod
    def get_dataset_root(rgb_file):
        "return the root path to all sequences"
        from pathlib import Path
        path = Path(rgb_file)
        return str(path.parents[2])

    @staticmethod
    def seqname2gender(seq_name:str):
        sub = seq_name.split('_')[1]
        return _sub_gender[sub]

ICAP_PATH = "/BS/xxie-6/static00/InterCap" # assume same root folder
date_seqs = {
    "Date01": lambda behave_path: behave_path + "/Date01_Sub01_backpack_back",
    "Date02": lambda behave_path: behave_path + "/Date02_Sub02_backpack_back",
    "Date03": lambda behave_path: behave_path + "/Date03_Sub03_backpack_back",
    "Date04": lambda behave_path: behave_path + "/Date04_Sub05_backpack",
    "Date05": lambda behave_path: behave_path + "/Date05_Sub05_backpack",
    "Date06": lambda behave_path: behave_path + "/Date06_Sub07_backpack_back",
    "Date07": lambda behave_path: behave_path + "/Date07_Sub04_backpack_back",
    "Date09": lambda procigen_path: procigen_path + "/Date09_Subxx_obj01_icap", # InterCap sequence synz
    # "ICapS01": ICAP_PATH + "/ICapS01_sub01_obj01_Seg_0",
    # "ICapS02": ICAP_PATH + "/ICapS02_sub01_obj08_Seg_0",
    # "ICapS03": ICAP_PATH + "/ICapS03_sub07_obj05_Seg_0",
}


_sub_gender = {
"Sub01": 'male',
"Sub02": 'male',
"Sub03": 'male',
"Sub04": 'male',
"Sub05": 'male',
"Sub06": 'female',
"Sub07": 'female',
"Sub08": 'female',
}