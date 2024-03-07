"""
Demo built with gradio
"""
import pickle as pkl
import sys, os
import os.path as osp
from typing import Iterable, Optional
from functools import partial

import trimesh
from torch.utils.data import DataLoader
import cv2
from accelerate import Accelerator
from tqdm import tqdm
from glob import glob

sys.path.append(os.getcwd())
import hydra
import torch
import numpy as np
import imageio
import gradio as gr
import plotly.graph_objs as go
import training_utils

from configs.structured import ProjectConfig
from demo import DemoRunner
from dataset.demo_dataset import DemoDataset


md_description="""
# HDM Interaction Reconstruction Demo
### Official Implementation of the paper \"Template Free Reconstruction of Human Object Interaction\", CVPR'24.
[Project Page](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/)|[Code](https://github.com/xiexh20/HDM)|[Dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.2VUEUS )|[Paper](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/paper-lowreso.pdf)


Upload your own human object interaction image and get full 3D reconstruction!

## Citation
```
@inproceedings{xie2023template_free,
    title = {Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
}
```
"""

def plot_points(colors, coords):
    """
    use plotly to visualize 3D point with colors
    """
    trace = go.Scatter3d(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode='markers',
                         marker=dict(
                             size=2,
                             color=colors
                         ))
    layout = go.Layout(
        scene=dict(
            xaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
            zaxis=dict(
                title="",
                showgrid=False,
                zeroline=False,
                showline=False,
                ticks='',
                showticklabels=False
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def inference(runner: DemoRunner, cfg: ProjectConfig, rgb, mask_hum, mask_obj, std_coverage, input_seed):
    """
    given user input, run inference
    :param runner:
    :param cfg:
    :param rgb: (h, w, 3), np array
    :param mask_hum: (h, w, 3), np array
    :param mask_obj: (h, w, 3), np array
    :param std_coverage: float value, used to estimate camera translation
    :param input_seed: random seed
    :return: path to the 3D reconstruction, and an interactive 3D figure for visualizing the point cloud
    """
    # Set random seed
    training_utils.set_seed(input_seed)

    data = DemoDataset([], (cfg.dataset.image_size, cfg.dataset.image_size),
                           std_coverage)
    batch = data.image2batch(rgb, mask_hum, mask_obj)

    out_stage1, out_stage2 = runner.forward_batch(batch, cfg)
    points = out_stage2.points_packed().cpu().numpy()
    colors = out_stage2.features_packed().cpu().numpy()
    fig = plot_points(colors, points)
    # save tmp point cloud
    outdir = './results'
    os.makedirs(outdir, exist_ok=True)
    trimesh.PointCloud(points, colors).export(outdir + f"/pred_std{std_coverage}_seed{input_seed}_stage2.ply")
    trimesh.PointCloud(out_stage1.points_packed().cpu().numpy(),
                       out_stage1.features_packed().cpu().numpy()).export(outdir + f"/pred_std{std_coverage}_seed{input_seed}_stage1.ply")
    return fig, outdir + f"/pred_std{std_coverage}_seed{input_seed}_stage2.ply"


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    # Setup model
    runner = DemoRunner(cfg)

    # Setup interface
    demo = gr.Blocks(title="HDM Interaction Reconstruction Demo")
    with demo:
        gr.Markdown(md_description)
        # Input data
        with gr.Row():
            input_rgb = gr.Image(label='Input RGB', type='numpy')
            input_mask_hum = gr.Image(label='Human mask', type='numpy')
        with gr.Row():
            input_mask_obj = gr.Image(label='Object mask', type='numpy')
            with gr.Column():
                # TODO: add hint for this value here
                input_std = gr.Number(label='Gaussian std coverage', value=3.5)
                input_seed = gr.Number(label='Random seed', value=42)
        # Output visualization
        with gr.Row():
            pc_plot = gr.Plot(label="Reconstructed point cloud")
            out_pc_download = gr.File(label="3D reconstruction for download") # this allows downloading

        # Control
        with gr.Row():
            button_recon = gr.Button("Reconstruct 3D", interactive=True, variant='secondary')
            button_recon.click(fn=partial(inference, runner, cfg),
                               inputs=[input_rgb, input_mask_hum, input_mask_obj, input_std, input_seed],
                               outputs=[pc_plot, out_pc_download])
    demo.launch()

if __name__ == '__main__':
    main()