from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch3d
import torch
from torch.utils.data import SequentialSampler
from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.data_loader_map_provider import \
    SequenceDataLoaderMapProvider
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.json_index_dataset_map_provider_v2 import (
    JsonIndexDatasetMapProviderV2, registry)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer.cameras import CamerasBase
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_meshes
import pickle as pkl
import os.path as osp

from configs.structured import CO3DConfig, DataloaderConfig, ProjectConfig, Optional
from .utils import DatasetMap



def get_dataset(cfg: ProjectConfig):
    
    if cfg.dataset.type == 'co3dv2':
        from .exclude_sequence import EXCLUDE_SEQUENCE, LOW_QUALITY_SEQUENCE
        dataset_cfg: CO3DConfig = cfg.dataset
        dataloader_cfg: DataloaderConfig = cfg.dataloader

        # Exclude bad and low-quality sequences, XH: why this is needed?
        exclude_sequence = []
        exclude_sequence.extend(EXCLUDE_SEQUENCE.get(dataset_cfg.category, []))
        exclude_sequence.extend(LOW_QUALITY_SEQUENCE.get(dataset_cfg.category, []))
        
        # Whether to load pointclouds
        kwargs = dict(
            remove_empty_masks=True,
            n_frames_per_sequence=1,
            load_point_clouds=True,
            max_points=dataset_cfg.max_points,
            image_height=dataset_cfg.image_size,
            image_width=dataset_cfg.image_size,
            mask_images=dataset_cfg.mask_images,
            exclude_sequence=exclude_sequence,
            pick_sequence=() if dataset_cfg.restrict_model_ids is None else dataset_cfg.restrict_model_ids,
        )

        # Get dataset mapper
        dataset_map_provider_type = registry.get(JsonIndexDatasetMapProviderV2, "JsonIndexDatasetMapProviderV2")
        expand_args_fields(dataset_map_provider_type)
        dataset_map_provider = dataset_map_provider_type(
            category=dataset_cfg.category,
            subset_name=dataset_cfg.subset_name,
            dataset_root=dataset_cfg.root,
            test_on_train=False,
            only_test_set=False,
            load_eval_batches=True,
            dataset_JsonIndexDataset_args=DictConfig(kwargs),
        )

        # Get datasets
        datasets = dataset_map_provider.get_dataset_map() # how to select specific frames??

        # PATCH BUG WITH POINT CLOUD LOCATIONS!
        for dataset in (datasets["train"], datasets["val"]):
            # print(dataset.seq_annots.items())
            for key, ann in dataset.seq_annots.items():
                correct_point_cloud_path = Path(dataset.dataset_root) / Path(*Path(ann.point_cloud.path).parts[-3:])
                assert correct_point_cloud_path.is_file(), correct_point_cloud_path
                ann.point_cloud.path = str(correct_point_cloud_path)

        # Get dataloader mapper
        data_loader_map_provider_type = registry.get(SequenceDataLoaderMapProvider, "SequenceDataLoaderMapProvider")
        expand_args_fields(data_loader_map_provider_type)
        data_loader_map_provider = data_loader_map_provider_type(
            batch_size=dataloader_cfg.batch_size,
            num_workers=dataloader_cfg.num_workers,
        )

        # QUICK HACK: Patch the train dataset because it is not used but it throws an error
        if (len(datasets['train']) == 0 and len(datasets[dataset_cfg.eval_split]) > 0 and 
                dataset_cfg.restrict_model_ids is not None and cfg.run.job == 'sample'):
            datasets = DatasetMap(train=datasets[dataset_cfg.eval_split], val=datasets[dataset_cfg.eval_split], 
                                  test=datasets[dataset_cfg.eval_split])
            # XH: why all eval split?
            print('Note: You used restrict_model_ids and there were no ids in the train set.')

        # Get dataloaders
        dataloaders = data_loader_map_provider.get_data_loader_map(datasets)
        dataloader_train = dataloaders['train']
        dataloader_val = dataloader_vis = dataloaders[dataset_cfg.eval_split]

        # Replace validation dataloader sampler with SequentialSampler
        # seems to be randomly sampled? with a fixed random seed? but one cannot control which image is being sampled??
        dataloader_val.batch_sampler.sampler = SequentialSampler(dataloader_val.batch_sampler.sampler.data_source)

        # Modify for accelerate
        dataloader_train.batch_sampler.drop_last = True
        dataloader_val.batch_sampler.drop_last = False
    elif cfg.dataset.type == 'shapenet_r2n2':
        # from ..configs.structured import ShapeNetR2N2Config
        from .r2n2_dataset import R2N2Sample
        dataset_cfg: ShapeNetR2N2Config = cfg.dataset
        # for k in dataset_cfg:
        #     print(k)
        datasets = [R2N2Sample(dataset_cfg.max_points, dataset_cfg.fix_sample,
                               dataset_cfg.image_size, cfg.augmentations,
                               s, dataset_cfg.shapenet_dir,
                               dataset_cfg.r2n2_dir, dataset_cfg.splits_file,
                               load_textures=False, return_all_views=True) for s in ['train', 'val', 'test']]
        dataloader_train = DataLoader(datasets[0], batch_size=cfg.dataloader.batch_size,
                                      collate_fn=collate_batched_meshes,
                                      num_workers=cfg.dataloader.num_workers, shuffle=True)
        dataloader_val = DataLoader(datasets[1], batch_size=cfg.dataloader.batch_size,
                                      collate_fn=collate_batched_meshes,
                                    num_workers=cfg.dataloader.num_workers, shuffle=False)
        dataloader_vis = DataLoader(datasets[2], batch_size=cfg.dataloader.batch_size,
                                      collate_fn=collate_batched_meshes,
                                    num_workers=cfg.dataloader.num_workers, shuffle=False)

    elif cfg.dataset.type in ['behave', 'behave-attn']:
        from .behave_dataset import BehaveDataset, BehaveTestOnly
        from .behave_paths import DataPaths
        from configs.structured import BehaveDatasetConfig
        from .behave_crossattn import BehaveCrossAttnDataset, BehaveCrossAttnTest

        dataset_cfg: BehaveDatasetConfig = cfg.dataset
        # pkl_file specifies the image paths without the root path, it can be downloaded from https://edmond.mpg.de/file.xhtml?fileId=251365&version=4.0
        pkl_file = cfg.dataset.split_file
        d = pkl.load(open(pkl_file, 'rb'))
        train_paths, val_paths = [osp.join(dataset_cfg.procigen_dir, x) for x in d['train']], [osp.join(dataset_cfg.behave_dir, x) for x in d['test']]

        # split validation/test paths to only consider the selected batches
        bs = cfg.dataloader.batch_size
        num_batches_total = int(np.ceil(len(val_paths)/cfg.dataloader.batch_size))
        end_idx = cfg.run.batch_end if cfg.run.batch_end is not None else num_batches_total
        val_paths = val_paths[cfg.run.batch_start*bs:end_idx*bs]

        if cfg.dataset.type == 'behave':
            train_type = BehaveDataset
            val_datatype = BehaveDataset if 'ntu' not in dataset_cfg.split_file else NTUDataset
        elif cfg.dataset.type == 'behave-test':
            train_type = BehaveDataset
            val_datatype = BehaveTestOnly
        elif cfg.dataset.type == 'behave-attn':
            train_type = BehaveCrossAttnDataset
            val_datatype = BehaveCrossAttnDataset
        elif cfg.dataset.type == 'behave-attn-test':
            train_type = BehaveCrossAttnDataset
            val_datatype = BehaveCrossAttnTest
        else:
            raise NotImplementedError

        dataset_train = train_type(train_paths, dataset_cfg.max_points, dataset_cfg.fix_sample,
                                   (dataset_cfg.image_size, dataset_cfg.image_size),
                                   split='train', sample_ratio_hum=dataset_cfg.sample_ratio_hum,
                                  normalize_type=dataset_cfg.normalize_type, smpl_type='gt',
                                  uniform_obj_sample=dataset_cfg.uniform_obj_sample,
                                  bkg_type=dataset_cfg.bkg_type,
                                  pred_binary=cfg.model.predict_binary,
                                  ho_segm_pred_path=cfg.dataset.ho_segm_pred_path,
                                  cam_noise_std=cfg.dataset.cam_noise_std,
                                  sep_same_crop=cfg.dataset.sep_same_crop,
                                  aug_blur=cfg.dataset.aug_blur,
                                  std_coverage=cfg.dataset.std_coverage,
                                   behave_path=dataset_cfg.behave_dir,
                                   procigen_path=dataset_cfg.procigen_dir)
        # we do cross validation, the validation set is a random subset of full test set
        dataset_val = val_datatype(val_paths, dataset_cfg.max_points, dataset_cfg.fix_sample,
                                      (dataset_cfg.image_size, dataset_cfg.image_size),
                                      split='val', sample_ratio_hum=dataset_cfg.sample_ratio_hum,
                                      normalize_type=dataset_cfg.normalize_type, smpl_type=dataset_cfg.smpl_type,
                                   test_transl_type=dataset_cfg.test_transl_type,
                                   uniform_obj_sample=dataset_cfg.uniform_obj_sample,
                                   bkg_type=dataset_cfg.bkg_type,
                                   pred_binary=cfg.model.predict_binary,
                                   ho_segm_pred_path=cfg.dataset.ho_segm_pred_path,
                                   sep_same_crop=cfg.dataset.sep_same_crop,
                                   std_coverage=cfg.dataset.std_coverage,
                                   behave_path=dataset_cfg.behave_dir,
                                   procigen_path=dataset_cfg.procigen_dir
                                   )
        dataloader_train = DataLoader(dataset_train, batch_size=cfg.dataloader.batch_size,
                                      collate_fn=collate_batched_meshes,
                                      num_workers=cfg.dataloader.num_workers, shuffle=True)
        shuffle = cfg.run.job == 'train'
        dataloader_val = DataLoader(dataset_val, batch_size=cfg.dataloader.batch_size,
                                    collate_fn=collate_batched_meshes,
                                    num_workers=cfg.dataloader.num_workers, shuffle=shuffle)
        dataloader_vis = DataLoader(dataset_val, batch_size=cfg.dataloader.batch_size,
                                    collate_fn=collate_batched_meshes,
                                    num_workers=cfg.dataloader.num_workers, shuffle=shuffle)
    elif cfg.dataset.type in ['shape']:
        from .shape_dataset import ShapeDataset
        from .behave_paths import DataPaths
        from configs.structured import ShapeDatasetConfig
        dataset_cfg: ShapeDatasetConfig = cfg.dataset

        train_paths, _ = DataPaths.load_splits(dataset_cfg.split_file, dataset_cfg.behave_dir)
        val_paths = train_paths # same as training, this is for overfitting
        # split validation paths to only consider the selected batches
        bs = cfg.dataloader.batch_size
        num_batches_total = int(np.ceil(len(val_paths) / cfg.dataloader.batch_size))
        end_idx = cfg.run.batch_end if cfg.run.batch_end is not None else num_batches_total
        # print(cfg.run.batch_end, cfg.run.batch_start, end_idx)
        val_paths = val_paths[cfg.run.batch_start * bs:end_idx * bs]

        dataset_train = ShapeDataset(train_paths, dataset_cfg.max_points, dataset_cfg.fix_sample,
                                   (dataset_cfg.image_size, dataset_cfg.image_size),
                                   split='train', )
        dataset_val = ShapeDataset(val_paths, dataset_cfg.max_points, dataset_cfg.fix_sample,
                                  (dataset_cfg.image_size, dataset_cfg.image_size),
                                  split='train', )
        dataloader_train = DataLoader(dataset_train, batch_size=cfg.dataloader.batch_size,
                                      collate_fn=collate_batched_meshes,
                                      num_workers=cfg.dataloader.num_workers, shuffle=True)
        shuffle = cfg.run.job == 'train'
        dataloader_val = DataLoader(dataset_val, batch_size=cfg.dataloader.batch_size,
                                    collate_fn=collate_batched_meshes,
                                    num_workers=cfg.dataloader.num_workers, shuffle=shuffle)
        dataloader_vis = DataLoader(dataset_val, batch_size=cfg.dataloader.batch_size,
                                    collate_fn=collate_batched_meshes,
                                    num_workers=cfg.dataloader.num_workers, shuffle=shuffle)
    else:
        raise NotImplementedError(cfg.dataset.type)
    
    return dataloader_train, dataloader_val, dataloader_vis
