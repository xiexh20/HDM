"""
Demo for template-free reconstruction

Usage: python demo.py model=ho-attn run.image_path=examples/017450/k1.color.jpg run.job=sample model.predict_binary=True dataset.std_coverage=3.0

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import pickle as pkl
import sys, os
import os.path as osp
from typing import Iterable, Optional

import cv2
from accelerate import Accelerator
from tqdm import tqdm
from glob import glob

sys.path.append(os.getcwd())
import hydra
import torch
import numpy as np
import imageio
from torch.utils.data import DataLoader
from pytorch3d.datasets import R2N2, collate_batched_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from pytorch3d.io import IO
import torchvision.transforms.functional as TVF
from huggingface_hub import hf_hub_download

import training_utils
from configs.structured import ProjectConfig
from dataset.demo_dataset import DemoDataset
from model import CrossAttenHODiffusionModel, ConditionalPCDiffusionSeparateSegm
from render.pyt3d_wrapper import PcloudRenderer


class DemoRunner:
    def __init__(self, cfg: ProjectConfig):
        cfg.model.model_name, cfg.model.predict_binary = 'pc2-diff-ho-sepsegm', True
        model_stage1 = ConditionalPCDiffusionSeparateSegm(**cfg.model)
        cfg.model.model_name, cfg.model.predict_binary = 'diff-ho-attn', False # stage 2 does not predict segmentation
        model_stage2 = CrossAttenHODiffusionModel(**cfg.model)

        # Load ckpt from hf
        if cfg.run.input_cls == 'general':
            ckpt_file1 = hf_hub_download("xiexh20/HDM-models", f'{cfg.run.stage1_name}.pth')
            self.load_checkpoint(ckpt_file1, model_stage1)
            ckpt_file2 = hf_hub_download("xiexh20/HDM-models", f'{cfg.run.stage2_name}.pth')
            self.load_checkpoint(ckpt_file2, model_stage2)
        else:
            assert cfg.run.input_cls in ['backpack', 'ball', 'bottle', 'box',
                                        'chair', 'skateboard', 'suitcase', 'table'], (f'no fine tuned models for '
                                                                                      f'class {cfg.run.input_cls}')
            self.reload_checkpoint(cfg.run.input_cls)

        self.model_stage1, self.model_stage2 = model_stage1, model_stage2
        self.model_stage1.eval()
        self.model_stage2.eval()
        self.model_stage1.to('cuda')
        self.model_stage2.to('cuda')

        self.cfg = cfg
        self.io_pc = IO()

        # For visualization
        self.renderer = PcloudRenderer(image_size=cfg.dataset.image_size, radius=0.0075)
        self.rend_size = cfg.dataset.image_size
        self.device = 'cuda'

    def load_checkpoint(self, ckpt_file1, model_stage1, device='cpu'):
        checkpoint = torch.load(ckpt_file1, map_location=device)
        state_dict, key = checkpoint['model'], 'model'
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            print('Removed "module." from checkpoint state dict')
        missing_keys, unexpected_keys = model_stage1.load_state_dict(state_dict, strict=False)
        print(f'Loaded model checkpoint {key} from {ckpt_file1}')
        if len(missing_keys):
            print(f' - Missing_keys: {missing_keys}')
        if len(unexpected_keys):
            print(f' - Unexpected_keys: {unexpected_keys}')

    def reload_checkpoint(self, cat_name):
        "load checkpoint of models fine tuned on specific categories"
        ckpt_file1 = hf_hub_download("xiexh20/HDM-models", f'{self.cfg.run.stage1_name}-{cat_name}.pth')
        self.load_checkpoint(ckpt_file1, self.model_stage1, device=self.device)
        ckpt_file2 = hf_hub_download("xiexh20/HDM-models", f'{self.cfg.run.stage2_name}-{cat_name}.pth')
        self.load_checkpoint(ckpt_file2, self.model_stage2, device=self.device)

    @torch.no_grad()
    def run(self):
        "simply run the demo on given images, and save the results"
        # Set random seed
        training_utils.set_seed(self.cfg.run.seed)

        outdir = osp.join(self.cfg.run.code_dir_abs, 'outputs/demo')
        os.makedirs(outdir, exist_ok=True)
        cfg = self.cfg

        # Init data
        image_files = sorted(glob(cfg.run.image_path))
        data = DemoDataset(image_files,
                           (cfg.dataset.image_size, cfg.dataset.image_size),
                           cfg.dataset.std_coverage)
        dataloader = DataLoader(data, batch_size=cfg.dataloader.batch_size,
                                collate_fn=collate_batched_meshes,
                                num_workers=1, shuffle=False)
        dataloader = dataloader
        progress_bar = tqdm(dataloader)
        for batch_idx, batch in enumerate(progress_bar):
            progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d}')

            for i in range(1):
                image_path = str(batch['image_path'])
                folder, fname = osp.basename(osp.dirname(image_path)), osp.splitext(osp.basename(image_path))[0]
                out_i = osp.join(outdir, folder)
                os.makedirs(out_i, exist_ok=True)
                # save RGB and masks
                cv2.imwrite(osp.join(out_i, 'k1.human_mask.png'), cv2.resize((batch['masks'][i][0].cpu().numpy()*255).astype(np.uint8), (800, 800)))
                cv2.imwrite(osp.join(out_i, 'k1.obj_mask.png'), cv2.resize((batch['masks'][i][1].cpu().numpy() * 255).astype(np.uint8), (800, 800)))
                print("Preprocessed images saved to", out_i)

            out_stage1, out_stage2 = self.forward_batch(batch, cfg)

            bs = len(out_stage1)
            camera_full = PerspectiveCameras(
                R=torch.stack(batch['R']),
                T=torch.stack(batch['T']),
                K=torch.stack(batch['K']),
                device='cuda',
                in_ndc=True)

            # save output
            for i in range(bs):
                image_path = str(batch['image_path'])
                folder, fname = osp.basename(osp.dirname(image_path)), osp.splitext(osp.basename(image_path))[0]
                out_i = osp.join(outdir, folder)
                os.makedirs(out_i, exist_ok=True)
                self.io_pc.save_pointcloud(data=out_stage1[i],
                                           path=osp.join(out_i, f'{fname}_stage1.ply'))
                self.io_pc.save_pointcloud(data=out_stage2[i],
                                           path=osp.join(out_i, f'{fname}_stage2.ply'))
                TVF.to_pil_image(batch['images'][i]).save(osp.join(out_i, f'{fname}_input.png'))

                # Save metadata as well
                metadata = dict(index=i,
                                camera=camera_full[i],
                                image_size_hw=batch['image_size_hw'][i],
                                image_path=batch['image_path'][i])
                torch.save(metadata, osp.join(out_i, f'{fname}_meta.pth'))

                # Visualize: compare stage 1 and stage 2 outputs side by side
                pc_comb = Pointclouds([out_stage1[i].points_packed(), out_stage2[i].points_packed()],
                                      features=[out_stage1[i].features_packed(), out_stage2[i].features_packed()])
                video_file = osp.join(out_i, f'{fname}_360view.mp4')
                video_writer = imageio.get_writer(video_file, format='FFMPEG', mode='I', fps=1)

                # first render front view
                rend_stage1, _ = self.renderer.render(out_stage1[i], camera_full[i], mode='mask')
                rend_stage2, _ = self.renderer.render(out_stage2[i], camera_full[i], mode='mask')
                comb = np.concatenate([batch['images'][i].permute(1, 2, 0).cpu().numpy(), rend_stage1, rend_stage2], 1)
                video_writer.append_data((comb*255).astype(np.uint8))

                # Render rotating view
                for azim in range(180, 180+360, 30):
                    R, T = look_at_view_transform(1.7, 0, azim, up=((0, -1, 0),), )
                    side_camera = PerspectiveCameras(image_size=((self.rend_size, self.rend_size),),
                                              device=self.device,
                                              R=R.repeat(2, 1, 1), T=T.repeat(2, 1),
                                              focal_length=self.rend_size * 1.5,
                                              principal_point=((self.rend_size / 2., self.rend_size / 2.),),
                                              in_ndc=False)
                    rend, mask = self.renderer.render(pc_comb, side_camera, mode='mask')

                    imgs = [batch['images'][i].permute(1, 2, 0).cpu().numpy()]
                    imgs.extend([rend[0], rend[1]])
                    video_writer.append_data((np.concatenate(imgs, 1)*255).astype(np.uint8))
                print(f"Visualization saved to {out_i}")

    @torch.no_grad()
    def forward_batch(self, batch, cfg):
        """
        forward one batch
        :param batch:
        :param cfg:
        :return: predicted point clouds of stage 1 and 2
        """
        camera_full = PerspectiveCameras(
            R=torch.stack(batch['R']),
            T=torch.stack(batch['T']),
            K=torch.stack(batch['K']),
            device='cuda',
            in_ndc=True)
        out_stage1 = self.model_stage1.forward_sample(num_points=cfg.dataset.max_points,
                                                      camera=camera_full,
                                                      image_rgb=torch.stack(batch['images']).to('cuda'),
                                                      mask=torch.stack(batch['masks']).to('cuda'),
                                                      scheduler=cfg.run.diffusion_scheduler,
                                                      num_inference_steps=cfg.run.num_inference_steps,
                                                      eta=cfg.model.ddim_eta,
                                                      )
        # segment and normalize human/object
        bs = len(out_stage1)
        pred_hum, pred_obj = [], []  # predicted human/object points
        cent_hum_pred, cent_obj_pred = [], []
        radius_hum_pred, radius_obj_pred = [], []
        T_hum, T_obj = [], []
        num_samples = int(cfg.dataset.max_points / 2)
        for i in range(bs):
            pc: Pointclouds = out_stage1[i]
            vc = pc.features_packed().cpu()  # (P, 3), human is light blue [0.1, 1.0, 1.0], object light green [0.5, 1.0, 0]
            points = pc.points_packed().cpu()  # (P, 3)
            mask_hum = vc[:, 2] > 0.5
            pc_hum, pc_obj = points[mask_hum], points[~mask_hum]
            # Up/Down-sample the points
            pc_obj = self.upsample_predicted_pc(num_samples, pc_obj)
            pc_hum = self.upsample_predicted_pc(num_samples, pc_hum)

            # Normalize
            cent_hum, cent_obj = torch.mean(pc_hum, 0, keepdim=True), torch.mean(pc_obj, 0, keepdim=True)
            scale_hum = torch.sqrt(torch.sum((pc_hum - cent_hum) ** 2, -1).max())
            scale_obj = torch.sqrt(torch.sum((pc_obj - cent_obj) ** 2, -1).max())
            pc_hum = (pc_hum - cent_hum) / (2 * scale_hum)
            pc_obj = (pc_obj - cent_obj) / (2 * scale_obj)
            # Also update camera parameters for separate human + object
            T_hum_scaled = (batch['T_ho'][i] + cent_hum.squeeze(0)) / (2 * scale_hum)
            T_obj_scaled = (batch['T_ho'][i] + cent_obj.squeeze(0)) / (2 * scale_obj)

            pred_hum.append(pc_hum)
            pred_obj.append(pc_obj)
            cent_hum_pred.append(cent_hum.squeeze(0))
            cent_obj_pred.append(cent_obj.squeeze(0))
            T_hum.append(T_hum_scaled * torch.tensor([-1, -1, 1]))  # apply opencv to pytorch3d transform: flip x and y
            T_obj.append(T_obj_scaled * torch.tensor([-1, -1, 1]))
            radius_hum_pred.append(scale_hum)
            radius_obj_pred.append(scale_obj)
        # Pack data into a new batch dict
        camera_hum = PerspectiveCameras(
            R=torch.stack(batch['R']),
            T=torch.stack(T_hum),
            K=torch.stack(batch['K_hum']),
            device='cuda',
            in_ndc=True
        )
        camera_obj = PerspectiveCameras(
            R=torch.stack(batch['R']),
            T=torch.stack(T_obj),
            K=torch.stack(batch['K_obj']),  # the camera should be human/object specific!!!
            device='cuda',
            in_ndc=True
        )
        # use pc from predicted
        pc_hum = Pointclouds([x.to('cuda') for x in pred_hum])
        pc_obj = Pointclouds([x.to('cuda') for x in pred_obj])
        # use center and radius from predicted
        cent_hum = torch.stack(cent_hum_pred, 0).to('cuda')
        cent_obj = torch.stack(cent_obj_pred, 0).to('cuda')  # B, 3
        radius_hum = torch.stack(radius_hum_pred, 0).to('cuda')  # B, 1
        radius_obj = torch.stack(radius_obj_pred, 0).to('cuda')
        out_stage2: Pointclouds = self.model_stage2.forward_sample(
            num_points=num_samples,
            camera=camera_hum,
            image_rgb=torch.stack(batch['images_hum'], 0).to('cuda'),
            mask=torch.stack(batch['masks_hum'], 0).to('cuda'),
            gt_pc=pc_hum,
            rgb_obj=torch.stack(batch['images_obj'], 0).to('cuda'),
            mask_obj=torch.stack(batch['masks_obj'], 0).to('cuda'),
            pc_obj=pc_obj,
            camera_obj=camera_obj,
            cent_hum=cent_hum,
            cent_obj=cent_obj,
            radius_hum=radius_hum.unsqueeze(-1),
            radius_obj=radius_obj.unsqueeze(-1),
            sample_from_interm=True,
            noise_step=cfg.run.sample_noise_step,
            scheduler=cfg.run.diffusion_scheduler,
            num_inference_steps=cfg.run.num_inference_steps,
            eta=cfg.model.ddim_eta,
        )
        return out_stage1, out_stage2

    def upsample_predicted_pc(self, num_samples, pc_obj):
        """
        Up/Downsample the points to given number
        :param num_samples: the target number
        :param pc_obj: (N, 3)
        :return: (num_samples, 3)
        """
        if len(pc_obj) > num_samples:
            ind_obj = np.random.choice(len(pc_obj), num_samples)
        else:
            ind_obj = np.concatenate([np.arange(len(pc_obj)), np.random.choice(len(pc_obj), num_samples - len(pc_obj))])
        pc_obj = pc_obj.clone()[torch.from_numpy(ind_obj).long().to(pc_obj.device)]
        return pc_obj


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    runner = DemoRunner(cfg)
    runner.run()


if __name__ == '__main__':
    main()