"""
Trainer for training and testing

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
"""
import datetime
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional
import os.path as osp

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TVF

import training_utils
import diffusion_utils
from dataset import get_dataset
from model import get_model
from configs.structured import ProjectConfig
from accelerate import DistributedDataParallelKwargs
from pytorch3d.structures import Pointclouds
import numpy as np
import pickle as pkl

from eval.chamfer_distance import chamfer_distance
torch.multiprocessing.set_sharing_strategy('file_system') # fix some bug in some servers

class Trainer(object):
    def __init__(self, cfg:ProjectConfig):
        # Accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True) # fix ddp problem
        accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu,
                                  gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
                                  kwargs_handlers=[ddp_kwargs])

        # Logging
        training_utils.setup_distributed_print(accelerator.is_main_process)
        if cfg.run.job == 'sample':
            cfg.logging.wandb = False
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.init(project=cfg.logging.wandb_project, name=cfg.run.name, job_type=cfg.run.job,
                       config=OmegaConf.to_container(cfg))
            wandb.run.log_code(root=hydra.utils.get_original_cwd(),
                               include_fn=lambda p: any(
                                   p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
                               exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
            cfg: ProjectConfig = DictConfig(wandb.config.as_dict())  # get the configs back from wandb for hyperparameter sweeps

        # Configuration
        # print(OmegaConf.to_yaml(cfg))
        print(f'Current working directory: {os.getcwd()}')

        # Set random seed
        training_utils.set_seed(cfg.run.seed)

        # Initialize model
        model = get_model(cfg)

        # Exponential moving average of model parameters
        if cfg.ema.use_ema:
            from torch_ema import ExponentialMovingAverage
            model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
            model_ema.to(accelerator.device)
            print('Initialized model EMA')
        else:
            model_ema = None
            print('Not using model EMA')
        self.model_ema = model_ema

        # Optimizer and scheduler
        optimizer = training_utils.get_optimizer(cfg, model, accelerator)
        scheduler = training_utils.get_scheduler(cfg, optimizer)

        # Resume from checkpoint and create the initial training state
        self.train_state: training_utils.TrainState = self.load_checkpoint(cfg, model, model_ema, optimizer, scheduler)

        # Datasets
        dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

        # Compute total training batch size
        self.total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

        # Setup. Note that this does not currently work with CO3D.
        model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis = accelerator.prepare(
            model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis)

        # for later use
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.dataloader_train, self.dataloader_val, self.dataloader_vis = dataloader_train, dataloader_val, dataloader_vis
        self.cfg = cfg
        self.accelerator = accelerator

        # additional data buffer
        self.loss_sep = None

    def load_checkpoint(self, cfg, model, model_ema, optimizer, scheduler):
        "load optimizer, model state, scheduler etc. "
        return training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler,
                                                     model_ema)

    def train(self, cfg:ProjectConfig):
        fscore_last, chamf_last = 0, 0.
        # Visualize before training
        if cfg.run.job == 'vis' or cfg.run.vis_before_training:
            fscores, chamfs = self.visualize(
                cfg=cfg,
                model=self.model,
                dataloader_vis=self.dataloader_vis,
                accelerator=self.accelerator,
                identifier=f'{self.train_state.step}',
                num_batches=1,
            )
            print(f"F-score={np.mean(fscores):.4f}, chamf={np.mean(chamfs):.4f}")
            fscore_last, chamf_last = np.mean(fscores), np.mean(chamfs)
            if cfg.run.job == 'vis':
                if cfg.logging.wandb and self.accelerator.is_main_process:
                    wandb.finish()
                    time.sleep(5)
                return

        self.print_info(cfg)

        # prepare for training
        train_state, optimizer, scheduler = self.train_state, self.optimizer, self.scheduler
        model, model_ema = self.model, self.model_ema
        accelerator = self.accelerator
        dataloader_train, dataloader_val, dataloader_vis = self.dataloader_train, self.dataloader_val, self.dataloader_vis


        # training loop
        while True:
            # Train progress bar
            log_header = f'Epoch: [{train_state.epoch}]'
            metric_logger = training_utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
            metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('fscore', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger.add_meter('chamf', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            metric_logger = self.add_log_item(metric_logger)
            progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq,
                                                                  header=log_header)

            # Train
            for i, batch in enumerate(progress_bar):
                if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
                model.train()

                # Gradient accumulation
                with accelerator.accumulate(model):

                    # Forward
                    loss = self.compute_loss(batch, model)

                    # Backward
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        if cfg.optimizer.clip_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                        grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                    # Step optimizer
                    optimizer.step()
                    optimizer.zero_grad()
                    if accelerator.sync_gradients:
                        scheduler.step()
                        train_state.step += 1

                    # Exit if loss was NaN
                    loss_value = loss.item()
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(90)

                # Gradient accumulation
                if accelerator.sync_gradients:
                    # Logging
                    log_dict = {
                        'lr': optimizer.param_groups[0]["lr"],
                        'step': train_state.step,
                        'train_loss': loss_value,
                        'grad_norm_clipped': grad_norm_clipped,
                        "fscore": fscore_last,
                        "chamf": chamf_last
                    }
                    log_dict = self.logging_addition(log_dict)
                    metric_logger.update(**log_dict)
                    if (
                            cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                        wandb.log(log_dict, step=train_state.step)

                    # Update EMA
                    if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                        model_ema.update(model.parameters())

                    # Save a checkpoint
                    if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                        self.save_checkpoint(accelerator, cfg, model, model_ema, optimizer, scheduler, train_state)

                    # Visualize
                    if (cfg.run.vis_freq > 0) and (train_state.step % cfg.run.vis_freq) == 0: # 5k steps
                        fscores, chamfs = self.visualize(
                            cfg=cfg,
                            model=model,
                            dataloader_vis=dataloader_vis,
                            accelerator=accelerator,
                            identifier=f'{train_state.step}',
                            num_batches=1,
                        )

                        fscore_last, chamf_last = np.mean(fscores), np.mean(chamfs)
                        print(f"updated F-score={fscore_last:.4f}, chamf={chamf_last:.4f}")

                    # End training after the desired number of steps/epochs
                    # or when lr is decreased to zero
                    if train_state.step >= cfg.run.max_steps or optimizer.param_groups[0]['lr'] < 1e-8:
                        print(f'Ending training at: {datetime.datetime.now()}')
                        print(f'Final train state: {train_state}')

                        wandb.finish()
                        time.sleep(5)
                        return

            # Epoch complete, log it and continue training
            train_state.epoch += 1

            # Gather stats from all processes
            metric_logger.synchronize_between_processes(device=self.accelerator.device)
            print(f'{log_header}  Average stats --', metric_logger)

    def print_info(self, cfg):
        # Info
        print(f'***** Starting training at {datetime.datetime.now()} *****')
        print(f'    Dataset train size: {len(self.dataloader_train.dataset):_}')
        print(f'    Dataset val size: {len(self.dataloader_train.dataset):_}')
        print(f'    Dataloader train size: {len(self.dataloader_train):_}')
        print(f'    Dataloader val size: {len(self.dataloader_val):_}')
        print(f'    Batch size per device = {cfg.dataloader.batch_size}')
        print(f'    Total train batch size (w. parallel, dist & accum) = {self.total_batch_size}')
        print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
        print(f'    Max training steps = {cfg.run.max_steps}')
        print(f'    Training state = {self.train_state}')

    def save_checkpoint(self, accelerator, cfg, model, model_ema, optimizer, scheduler, train_state):
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        checkpoint_dict = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': train_state.epoch,
            'step': train_state.step,
            'best_val': train_state.best_val,
            'model_ema': model_ema.state_dict() if model_ema else {},
            'cfg': cfg
        }
        checkpoint_path = 'checkpoint-latest.pth'
        # check if checkpoint exist
        if osp.isfile(checkpoint_path):
            ckpt_old = torch.load(checkpoint_path)
            if train_state.step > 60000 and train_state.step %5000 ==0 :
                # keep old checkpoints that have been trained for more than 100k steps \
                old_step = ckpt_old['step']
                newfile = checkpoint_path.replace('-latest.pth', f'-step-{old_step:07d}.pth')
                torch.save(ckpt_old, newfile)

            accelerator.save(checkpoint_dict, checkpoint_path)
            print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

        else:
            accelerator.save(checkpoint_dict, checkpoint_path)
            print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')

    def compute_loss(self, batch, model):
        return model(batch, mode='train')

    def run_sample(self, cfg:ProjectConfig):
        # Whether or not to use EMA parameters for sampling
        if cfg.run.sample_from_ema:
            assert self.model_ema is not None
            self.model_ema.to(self.accelerator.device)
            sample_context = self.model_ema.average_parameters
        else:
            sample_context = nullcontext
        # Sample
        with sample_context():
            self.sample(
                cfg=cfg,
                model=self.model,
                dataloader=self.dataloader_val,
                accelerator=self.accelerator,
                output_dir=cfg.run.save_name
            )
        if cfg.logging.wandb and self.accelerator.is_main_process:
            wandb.finish()
        print('all done')
        time.sleep(5)

    def logging_addition(self, log_dict:dict):
        return log_dict

    def add_log_item(self, metric_logger):
        return metric_logger

    def is_done(self, batch, output_dir: str):
        "check if this batch is done"
        bs = self.get_batch_size(batch)
        filename = '{name}.{ext}'
        filestr = str(output_dir / '{dir}' / '{category}' / filename)
        for i in range(bs):
            pred_file = filestr.format(dir='pred', category=self.get_seq_category(batch, i), name=self.get_seq_name(batch, i), ext='ply')
            if not os.path.isfile(pred_file):
                return False
        return True

    @torch.no_grad()
    def sample(self, cfg: ProjectConfig,
                model: torch.nn.Module,
                dataloader: Iterable,
                accelerator: Accelerator,
                output_dir: str = 'sample',):
        from pytorch3d.io import IO
        from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
        from pytorch3d.structures import Pointclouds
        from tqdm import tqdm

        # Eval mode
        model.eval()
        progress_bar: Iterable[FrameData] = tqdm(dataloader, disable=(not accelerator.is_main_process))

        # Output dir
        output_dir: Path = Path(output_dir)

        # PyTorch3D IO
        # io = IO()

        end_idx = cfg.run.batch_end if cfg.run.batch_end is not None else len(dataloader)
        # Visualize
        for batch_idx, batch in enumerate(progress_bar):
            progress_bar.set_description(f'Processing batch {batch_idx:4d} / {len(dataloader):4d}')
            if cfg.run.num_sample_batches is not None and batch_idx >= cfg.run.num_sample_batches:
                break

            # only run for specific batches
            if cfg.dataset.type == 'shapenet_r2n2':
                if batch_idx < cfg.run.batch_start:
                    print(f"Skipped batch {batch_idx}.")
                    continue
                if batch_idx >= end_idx:
                    break

            # import pdb;
            # pdb.set_trace() # batch keys:
            # print([k for k in batch])

            # for debug: save sampled frames
            filename = '{name}.{ext}'
            filestr = str(output_dir / '{dir}' / '{category}' / filename)
            sequence_category = self.get_seq_category(batch, 0) # TODO: replace for different dataset

            file = filestr.format(dir='images', category=sequence_category, name=f"batch_{batch_idx:02d}", ext='json')
            os.makedirs(os.path.dirname(file), exist_ok=True)
            json.dump(batch['image_path'], open(file, 'w'))
            print("sequence:", sequence_category, 'first image:', batch['image_path'][0])
            # continue

            # Optionally produce multiple samples for each point cloud
            for sample_idx in range(cfg.run.num_samples):
                if self.is_done(batch, output_dir) and not cfg.run.redo:
                    print(f"batch {batch_idx} already done, skipped")
                    continue

                # Filestring
                filename = f'{{name}}-{sample_idx}.{{ext}}' if cfg.run.num_samples > 1 else '{name}.{ext}'
                filestr = str(output_dir / '{dir}' / '{category}' / filename)

                # Sample
                w_joint = 0 if cfg.model.model_name != 'diff-comb' else cfg.model.model_joint_weight
                w_sep = 0 if cfg.model.model_name != 'diff-comb' else cfg.model.model_sep_weight
                output, all_outputs = model(batch, mode=cfg.run.sample_mode,
                                            return_sample_every_n_steps=10,
                                            scheduler=cfg.run.diffusion_scheduler,
                                            num_inference_steps=cfg.run.num_inference_steps,
                                            disable_tqdm=(not accelerator.is_main_process),
                                            noise_step=cfg.run.sample_noise_step,
                                            w_joint=w_joint, w_sep=w_sep, # for combined diffusion model
                                            eta=cfg.model.ddim_eta,
                                            )
                output: Pointclouds
                all_outputs: List[Pointclouds]  # list of B Pointclouds, each with a batch size of return_sample_every_n_steps

                # Save individual samples
                for i in range(len(output)):
                    sequence_name = self.get_seq_name(batch, i)
                    sequence_category = self.get_seq_category(batch, i)

                    (output_dir / 'gt' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'pred' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'images' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'metadata' / sequence_category).mkdir(exist_ok=True, parents=True)
                    (output_dir / 'evolutions' / sequence_category).mkdir(exist_ok=True, parents=True)

                    # Save ground truth
                    self.save_pclouds(batch, filestr, i, output, sequence_category, sequence_name, cfg.run.sample_save_gt)

                    # Save input images
                    filename = filestr.format(dir='images', category=sequence_category, name=sequence_name, ext='png')
                    TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)
                    # self.save_input_image(batch, filename, i)
                    # print('saved to', filename)

                    # Save camera
                    filename = filestr.format(dir='metadata', category=sequence_category, name=sequence_name, ext='pth')
                    metadata = self.get_metadata(batch, i)
                    torch.save(metadata, filename)

                    # Save evolutions
                    if cfg.run.sample_save_evolutions:
                        torch.save(all_outputs[i], filestr.format(dir='evolutions', category=sequence_category,
                                                                  name=sequence_name, ext='pth'))

        print('Saved samples to: ')
        print(output_dir.absolute())

    def save_pclouds(self, batch, filestr, i, output, sequence_category, sequence_name, save_gt=True):
        from pytorch3d.io import IO
        io = IO()
        if save_gt:
            io.save_pointcloud(data=self.get_gt_pclouds(batch, i), path=filestr.format(dir='gt',
                                                                                       category=sequence_category,
                                                                                       name=sequence_name,
                                                                                       ext='ply'))
        # Save generation
        io.save_pointcloud(data=output[i], path=filestr.format(dir='pred',
                                                               category=sequence_category,
                                                               name=sequence_name, ext='ply'))

        # save binary segmentation if presented
        pc: Pointclouds = output[i]
        if pc.features_packed() is not None:
            # with segmentation color, save segmentation results
            assert len(pc.features_list()) == 1
            vc = pc.features_packed()  # (P, 3), human is light blue [0.1, 1.0, 1.0], object light green [0.5, 1.0, 0]
            points = pc.points_packed()  # (P, 3)
            mask_hum = vc[:, 2] > 0.5
            pc_hum, pc_obj = points[mask_hum], points[~mask_hum]
            assert len(pc_hum) > 10, f"Only {len(pc_hum)} human points found in {batch['image_path'][i]}!"
            assert len(pc_obj) > 10, f"Only {len(pc_obj)} object points found in {batch['image_path'][i]}!"
            transl_hum = torch.mean(pc_hum, 0)
            transl_obj = torch.mean(pc_obj, 0)
            scale_hum = torch.sqrt(torch.max(torch.sum((pc_hum - transl_hum) ** 2, -1))).cpu().numpy()
            scale_obj = torch.sqrt(torch.max(torch.sum((pc_obj - transl_obj) ** 2, -1))).cpu().numpy()
            out = {
                "pred_trans": torch.cat([transl_hum, transl_obj], 0).cpu().numpy(),
                "pred_scale": np.array([scale_hum, scale_obj])
            }
            outfile = filestr.format(dir='pred', category=sequence_category, name=sequence_name, ext='pkl')
            # print(f"{torch.sum(mask_hum)}/{len(points)} human points")
            pkl.dump(out, open(outfile, 'wb'))

            # save gt
            pc_gt = self.get_gt_pclouds(batch, i)
            points = pc_gt.points_packed()
            L = len(points)
            pc_hum, pc_obj = points[:L // 2], points[L // 2:]
            transl_hum = torch.mean(pc_hum, 0)
            transl_obj = torch.mean(pc_obj, 0)
            scale_hum = torch.sqrt(torch.max(torch.sum((pc_hum - transl_hum) ** 2, -1))).cpu().numpy()
            scale_obj = torch.sqrt(torch.max(torch.sum((pc_obj - transl_obj) ** 2, -1))).cpu().numpy()
            out = {
                "gt_trans": torch.cat([transl_hum, transl_obj], 0).cpu().numpy(),
                "gt_scale": np.array([scale_hum, scale_obj]),
                "num_smpl": L // 2,
                "samples": points.cpu().numpy()
            }
            outfile = filestr.format(dir='gt', category=sequence_category, name=sequence_name, ext='pkl')
            # print(f"{torch.sum(mask_hum)}/{len(points)} human points")
            pkl.dump(out, open(outfile, 'wb'))

    def get_metadata(self, batch, i):
        metadata = dict(index=i, sequence_name=batch.sequence_name,
                        sequence_category=batch.sequence_category,
                        frame_timestamp=batch.frame_timestamp, camera=batch.camera,
                        image_size_hw=batch.image_size_hw,
                        image_path=batch.image_path, depth_path=batch.depth_path, mask_path=batch.mask_path,
                        bbox_xywh=batch.bbox_xywh, crop_bbox_xywh=batch.crop_bbox_xywh,
                        sequence_point_cloud_path=batch.sequence_point_cloud_path, meta=batch.meta)
        return metadata

    def save_input_image(self, batch, filename, i):
        TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)

    def get_input_image(self, batch, i):
        return batch.image_rgb[i]

    def get_seq_name(self, batch, i):
        sequence_name = batch.sequence_name[i]
        return sequence_name

    def get_seq_category(self, batch, ind=0):
        sequence_category = batch.sequence_category[ind]
        return sequence_category

    def get_batch_size(self, batch):
        return len(batch.image_rgb)

    @torch.no_grad()
    def visualize(
            self,
            cfg: ProjectConfig,
            model: torch.nn.Module,
            dataloader_vis: Iterable,
            accelerator: Accelerator,
            identifier: str = '',
            num_batches: Optional[int] = None,
            output_dir: str = 'vis',
    ):
        from pytorch3d.vis.plotly_vis import plot_scene
        from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
        from pytorch3d.structures import Pointclouds

        # Eval mode
        model.eval()
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        progress_bar: Iterable[FrameData] = metric_logger.log_every(dataloader_vis, cfg.run.print_step_freq, "Vis")

        # Output dir
        output_dir: Path = Path(output_dir)
        (output_dir / 'raw').mkdir(exist_ok=True, parents=True)
        (output_dir / 'pointclouds').mkdir(exist_ok=True, parents=True)
        (output_dir / 'images').mkdir(exist_ok=True, parents=True)
        (output_dir / 'videos').mkdir(exist_ok=True, parents=True)
        (output_dir / 'evolutions').mkdir(exist_ok=True, parents=True)
        (output_dir / 'metadata').mkdir(exist_ok=True, parents=True)

        # Visualize
        wandb_log_dict = {}
        fscores, chamfs = [], []
        for batch_idx, batch in enumerate(progress_bar):
            if num_batches is not None and batch_idx >= num_batches:
                break

            # Sample
            output, all_outputs = model(batch, mode='sample', return_sample_every_n_steps=100,
                                        num_inference_steps=cfg.run.num_inference_steps,
                                        disable_tqdm=(not accelerator.is_main_process))
            output: Pointclouds
            all_outputs: List[Pointclouds]  # list of B Pointclouds, each with a batch size of return_sample_every_n_steps

            # Filenames
            filestr = str(
                output_dir / '{dir}' / f'p-{accelerator.process_index}-b-{batch_idx}-s-{{i:02d}}-{{name}}-{identifier}.{{ext}}')
            filestr_wandb = f'{{dir}}/b-{batch_idx}-{{name}}-s-{{i:02d}}-{{name}}' # identifier=init

            # Not saving raw samples are they are too big
            filename = filestr.format(dir='raw', name='raw', i=0, ext='pth')
            # torch.save({'output': output, 'all_outputs': all_outputs, 'batch': batch}, filename)

            # Save metadata
            metadata = diffusion_utils.get_metadata(batch)
            filename = filestr.format(dir='metadata', name='metadata', i=0, ext='txt')
            Path(filename).write_text(metadata)

            # Save individual samples
            for i in range(len(output)):
                camera, gt_pointcloud = self.preprocess_gt(batch, i) # this should be updated for different datasets

                pred_pointcloud = output[i]
                pred_all_pointclouds = all_outputs[i]

                # Plot using plotly and pytorch3d
                fig = plot_scene({
                    'Pred': {'pointcloud': pred_pointcloud},
                    'GT': {'pointcloud': gt_pointcloud},
                }, ncols=2, viewpoint_cameras=camera, pointcloud_max_points=16_384)

                # Save plot, don't save html, it is too large
                # filename = filestr.format(dir='pointclouds', name='pointclouds', i=i, ext='html')
                # fig.write_html(filename)

                # Add to W&B, don't save html, this is too large
                # filename_wandb = filestr_wandb.format(dir='pointclouds', name='pointclouds', i=i)
                # wandb_log_dict[filename_wandb] = wandb.Html(open(filename), inject=False)

                # Save input images
                filename = filestr.format(dir='images', name='image_rgb', i=i, ext='png')
                TVF.to_pil_image(self.get_input_image(batch, i)).save(filename)

                # Add to W&B
                filename_wandb = filestr_wandb.format(dir='images', name='image_rgb', i=i)
                wandb_log_dict[filename_wandb] = wandb.Image(filename)

                # TODO: compute evaluation error here
                fscore, cd = self.compute_errors(gt_pointcloud, pred_pointcloud)
                fscores.append(fscore)
                chamfs.append(cd)

                # Loop
                for name, pointcloud in (('gt', gt_pointcloud), ('pred', pred_pointcloud)):
                    # Render gt/pred point cloud from given view
                    # these images are saved to vis/images/
                    filename_image = filestr.format(dir='images', name=name, i=i, ext='png')
                    filename_image_wandb = filestr_wandb.format(dir='images', name=name, i=i)
                    diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud,
                                                                         output_file_image=filename_image,
                                                                         cameras=camera,
                                                                         scale_factor=cfg.model.scale_factor)
                    wandb_log_dict[filename_image_wandb] = wandb.Image(filename_image)

                    # Render gt/pred point cloud from rotating view
                    filename_video = filestr.format(dir='videos', name=name, i=i, ext='mp4')
                    filename_video_wandb = filestr_wandb.format(dir='videos', name=name, i=i)
                    diffusion_utils.visualize_pointcloud_batch_pytorch3d(pointclouds=pointcloud,
                                                                         output_file_video=filename_video,
                                                                         num_frames=30,
                                                                         scale_factor=cfg.model.scale_factor)
                    wandb_log_dict[filename_video_wandb] = wandb.Video(filename_video)

                # Render point cloud diffusion evolution
                filename_evo = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
                filename_evo_wandb = filestr.format(dir='evolutions', name='evolutions', i=i, ext='mp4')
                diffusion_utils.visualize_pointcloud_evolution_pytorch3d(
                    pointclouds=pred_all_pointclouds, output_file_video=filename_evo, camera=camera)
                wandb_log_dict[filename_evo_wandb] = wandb.Video(filename_evo)

        # Save to W&B
        if cfg.logging.wandb and accelerator.is_local_main_process:
            wandb.log(wandb_log_dict, commit=False)

        print('Saved visualizations to: ')
        print(output_dir.absolute())
        return fscores, chamfs

    def preprocess_gt(self, batch, i):
        "preprocess for sampling"
        camera = self.get_camera(batch, i)
        gt_pointcloud = self.get_gt_pclouds(batch, i)
        return camera, gt_pointcloud

    def get_camera(self, batch, i):
        return batch.camera[i]

    def get_gt_pclouds(self, batch, i):
        gt_pointcloud = batch.sequence_point_cloud[i]
        return gt_pointcloud

    def compute_errors(self, gt:Pointclouds, pred:Pointclouds, thres:float=0.01):
        """
        compute F-score, CD between gt and prediction
        :param gt:
        :param pred:
        :return:
        """
        assert len(gt.points_list()) == 1, f'found gt points of batch size {len(gt.points_list())}'
        assert len(pred.points_list()) == 1, f'found predicted points of batch size {len(pred.points_list())}'
        gt = gt.points_packed().cpu().numpy()
        pred = pred.points_packed().cpu().numpy()

        if np.any(np.isnan(gt)) or np.any(np.isnan(pred)):
            print("Warning: found NaN values in predicted points!")
            return 0, 100.

        chamf, fscore = self.compute_fscore_chamf(gt, pred, thres)
        return fscore, chamf

    def compute_fscore_chamf(self, gt, pred, thres):
        """
        compute fscore with numpy array, gt and pred are both (N, 3)
        """
        chamf, d1, d2 = chamfer_distance(gt, pred, ret_intermediate=True)
        recall = float(sum(d < thres for d in d2)) / float(len(d2))
        precision = float(sum(d < thres for d in d1)) / float(len(d1))
        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
        return chamf, fscore