# Hierarchical Diffusion Model (CVPR'24)
Official implementation for the HDM model of the CVPR24 paper: Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation

[Project Page](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/) | [ProciGen Dataset](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.2VUEUS ) | [Paper](https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/paper-lowreso.pdf) | <a href="https://huggingface.co/spaces/xiexh20/HDM-interaction-recon"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange'></a><br></br>

<p align="left">
<img src="https://virtualhumans.mpi-inf.mpg.de/procigen-hdm/teaser_full_width.png" alt="teaser" width="70%"/>
</p>

## For template based reconstruction please see: [CHORE](https://github.com/xiexh20/CHORE), [VisTracker](https://github.com/xiexh20/VisTracker). 

## Contents 
1. [Dependencies](#dependencies)
3. [Run demo](#run-demo)
4. [Training](#training)
5. [Evaluation](#evaluation)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)
7. [License](#license)

### TODO List
- [x] Hugging face and Gradio demo.
- [ ] Google Colab demo.
- [ ] Training and inference. 
- [ ] Evaluation code. 

## Dependencies
The code is tested on `torch=1.12.1+cu113, cuda11.3, debian11`. We recommend using anaconda environment:
```shell
conda create -n hdm python=3.8
conda activate hdm 
```
Required packages can be installed by:
```shell
pip install -r pre-requirements.txt # Install pytorch and others
pip install -r requirements.txt     # Install pytorch3d from source
```
In case pytorch3d compilation failed, you can tried to install prebuilt wheels. In this case, pytorch should also be reinstalled. 
The following combination of torch and pytorch3d has been tested to be compatible:
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

## Run demo
**Hugging face demo:** <a href="https://huggingface.co/spaces/xiexh20/HDM-interaction-recon"  style='padding-left: 0.5rem;'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange'></a><br></br>
**Google Colab:** coming soon.

Run our gradio demo on your own machine:
```shell
python app.py
```
In case of a headless remote server, you can run `python app.py run.share=True` to create a temporal public url which you can access the webpage in laptop browser. 

Alternatively, you can run the demo given image path:
```shell
python demo.py run.image_path=<RGB image path> dataset.std_coverage=<optional, a value between 3.0 and 3.8> 
```
For example: `python demo.py run.image_path=$PWD/examples/017450/k1.color.jpg dataset.std_coverage=3.0`

[//]: # (### Hugging face demo: [HDM 🤗]&#40;&#41;)



## Training
### Coming soon

## Evaluation
### Coming soon

## Citation
If you use the code, please cite: 
```
@inproceedings{xie2023template_free,
    title = {Template Free Reconstruction of Human-object Interaction with Procedural Interaction Generation},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
}
```

## Acknowledgements
This project leverages the following excellent works, we thank the authors for open-sourcing their code: 

* The [PyTorch3D](https://github.com/facebookresearch/pytorch3d) library. 
* The [diffusers](https://github.com/huggingface/diffusers) library. 
* The [pc2](https://github.com/lukemelas/projection-conditioned-point-cloud-diffusion/tree/main) project.

## License
Please see [LICENSE](./LICENSE).
