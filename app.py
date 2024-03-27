"""
Demo built with gradio

Author: Xianghui Xie
Date: March 27, 2024
Cite: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation
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
import traceback

from configs.structured import ProjectConfig
from demo import DemoRunner
from dataset.demo_dataset import DemoDataset


md_description="""
# HDM Interaction Reconstruction Demo
### Official Demo of the paper \"Template Free Reconstruction of Human Object Interaction\", CVPR'24.
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
citation_str = """
## Citation
```
@inproceedings{xie2023template_free,
    title = {Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
}
"""

html_str = """
<h2 style="text-align:center; color:#10768c">HDM Demo: Upload your own human object interaction image and get full 3D reconstruction!</h2>
<h3 style="text-align:center; color:#10768c">Official Demo of "Template Free Reconstruction of Human Object Interaction with Procedural Generation", CVPR'24. </h3>
<h3 style="text-align:center; color:#10768c"><a href="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/" target="_blank">Project Page</a> | 
    <a href="https://github.com/xiexh20/HDM" target="_blank">Code</a> | 
    <a href="https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.2VUEUS" target="_blank">ProciGen Dataset</a> | 
    <a href="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/paper-lowreso.pdf" target="_blank">Paper</a>    
</h3>

<p style="text-align:left; color:#10768c">Instruction:
<ol>
    <li>Upload an RGB image of human object interaction.</li>
    <li>Upload the mask for the human and object that you want to reconstruct. You can use these methods to obtain the masks: 
                <a href="https://segment-anything.com/demo" target="_blank">SAM</a>, 
                <a href="https://huggingface.co/spaces/sam-hq-team/sam-hq" target="_blank">SAM-HQ</a>,
                <a href="https://huggingface.co/spaces/An-619/FastSAM" target="_blank">FastSAM</a>.</li>
    <li>Click `Start Reconstruction` to start.</li>
    <li>You can view the result at `Reconstructed point cloud` and download the point cloud at `download results`. </li>
</ol>
Alternatively, you can click one of the examples below and start reconstruction. 
</p>
<p>More example results can be found in our <a href="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/" target="_blank">Project Page</a>.</p>
<p>Have fun! </p>
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


def inference(runner: DemoRunner, cfg: ProjectConfig, rgb, mask_hum, mask_obj, std_coverage, input_seed, input_cls):
    """
    given user input, run inference
    :param runner:
    :param cfg:
    :param rgb: (h, w, 3), np array
    :param mask_hum: (h, w, 3), np array
    :param mask_obj: (h, w, 3), np array
    :param std_coverage: float value, used to estimate camera translation
    :param input_seed: random seed
    :param input_cls: the object category of the input image
    :return: path to the 3D reconstruction, and an interactive 3D figure for visualizing the point cloud
    """
    log = ""
    try:
        # Set random seed
        training_utils.set_seed(int(input_seed))

        data = DemoDataset([], (cfg.dataset.image_size, cfg.dataset.image_size),
                           std_coverage)
        batch = data.image2batch(rgb, mask_hum, mask_obj)

        if input_cls != 'general':
            log += f"Reloading fine-tuned checkpoint of category {input_cls}\n"
            runner.reload_checkpoint(input_cls)

        out_stage1, out_stage2 = runner.forward_batch(batch, cfg)
        points = out_stage2.points_packed().cpu().numpy()
        colors = out_stage2.features_packed().cpu().numpy()
        fig = plot_points(colors, points)
        # save tmp point cloud
        outdir = './results'
        os.makedirs(outdir, exist_ok=True)
        trimesh.PointCloud(points, colors).export(outdir + f"/pred_std{std_coverage}_seed{input_seed}_stage2_{input_cls}.ply")
        trimesh.PointCloud(out_stage1.points_packed().cpu().numpy(),
                           out_stage1.features_packed().cpu().numpy()).export(
            outdir + f"/pred_std{std_coverage}_seed{input_seed}_stage1_{input_cls}.ply")
        log += 'Successfully reconstructed the image.'
        outfile = outdir + f"/pred_std{std_coverage}_seed{input_seed}_stage2_{input_cls}.ply"
    except Exception as e:
        log = traceback.format_exc()
        fig, outfile = None, None

    return fig, outfile, log


@hydra.main(config_path='configs', config_name='configs', version_base='1.1')
def main(cfg: ProjectConfig):
    # Setup model
    runner = DemoRunner(cfg)

    # Setup interface
    demo = gr.Blocks(title="HDM Interaction Reconstruction Demo")
    with demo:
        gr.HTML(html_str)
        # Input data
        with gr.Row():
            input_rgb = gr.Image(label='Input RGB', type='numpy')
            input_mask_hum = gr.Image(label='Human mask', type='numpy')
        with gr.Row():
            input_mask_obj = gr.Image(label='Object mask', type='numpy')
            with gr.Column():
                input_std = gr.Number(label='Gaussian std coverage', value=3.5,
                                      info="This value is used to estimate camera translation to project the points."
                                           "The larger value, the camera is farther away. It is category-dependent. "
                                           "We empirically found these values are suitable: backpack-3.5, ball-3.0, bottle-3.0,"
                                           "box-3.5, chair-3.8, skateboard-3.0, suitcase-3.2, table-3.5. "
                                           "If you are not sure, 3.5 is a good start point.")
                input_cls = gr.Dropdown(label='Object category',
                                        info='We fine tuned the model for some specific categories. '
                                             'Reconstructing using these models should lead to better result '
                                             'for these specific categories. Simply select the category that '
                                             'fits the object from input image.',
                                        choices=['general', 'backpack', 'ball', 'bottle', 'box',
                                                 'chair', 'skateboard', 'suitcase', 'table'],
                                        value='general')
                input_seed = gr.Number(label='Random seed', value=42)
        # Output visualization
        with gr.Row():
            pc_plot = gr.Plot(label="Reconstructed point cloud")
            out_pc_download = gr.File(label="Download results") # this allows downloading
        with gr.Row():
            out_log = gr.TextArea(label='Output log')


        gr.HTML("""<br/>""")
        # Control
        with gr.Row():
            button_recon = gr.Button("Start Reconstruction", interactive=True, variant='secondary')
            button_recon.click(fn=partial(inference, runner, cfg),
                               inputs=[input_rgb, input_mask_hum, input_mask_obj, input_std, input_seed, input_cls],
                               outputs=[pc_plot, out_pc_download, out_log])
        gr.HTML("""<br/>""")
        # Example input
        example_dir = cfg.run.code_dir_abs+"/examples"
        rgb, ps, obj = 'k1.color.jpg', 'k1.person_mask.png', 'k1.obj_rend_mask.png'
        example_images = gr.Examples([
            [f"{example_dir}/017450/{rgb}", f"{example_dir}/017450/{ps}", f"{example_dir}/017450/{obj}", 3.0, 42, 'skateboard'],
            [f"{example_dir}/205904/{rgb}", f"{example_dir}/205904/{ps}", f"{example_dir}/205904/{obj}", 3.2, 42, 'suitcase'],
            [f"{example_dir}/066241/{rgb}", f"{example_dir}/066241/{ps}", f"{example_dir}/066241/{obj}", 3.5, 42, 'backpack'],
            [f"{example_dir}/053431/{rgb}", f"{example_dir}/053431/{ps}", f"{example_dir}/053431/{obj}", 3.8, 42, 'chair'],
            [f"{example_dir}/158107/{rgb}", f"{example_dir}/158107/{ps}", f"{example_dir}/158107/{obj}", 3.8, 42, 'chair'],

        ], inputs=[input_rgb, input_mask_hum, input_mask_obj, input_std, input_seed, input_cls],)

        gr.Markdown(citation_str)

    # demo.launch(share=True)
    # Enabling queue for runtime>60s, see: https://github.com/tloen/alpaca-lora/issues/60#issuecomment-1510006062
    demo.queue().launch(share=cfg.run.share)

if __name__ == '__main__':
    main()
