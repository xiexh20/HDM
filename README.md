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
- [x] Training and inference. 
- [x] Evaluation code. 

### Updates
- June 12, 2024. Add support for using DDIM for inference, the speed is now 10x faster, with minimum affects on performance! 

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

**Gradio demo on your own machine**:

Simply run with:  
```shell
# option 1: original DDPM inference
python app.py 

# option 2: 10x faster inference with DDIM:
python app.py run.diffusion_scheduler=ddim run.num_inference_steps=100
```
In case of a headless remote server, you can run `python app.py run.share=True` to create a temporal public url which you can access the webpage in laptop browser. 

**Command line demo**:

Alternatively, you can run the demo given image path:
```shell
python demo.py run.image_path=<RGB image path> dataset.std_coverage=<optional, a value between 3.0 and 3.8> 
```
For example: `python demo.py run.image_path=$PWD/examples/017450/k1.color.jpg dataset.std_coverage=3.0`

Similar to running `app.py`, you can speed up inference by appending `run.diffusion_scheduler=ddim run.num_inference_steps=100` to the end of the command. 

[//]: # (### Hugging face demo: [HDM 🤗]&#40;&#41;)



## Training
#### Download data
We mainly train our model on synthetic ProciGen dataset and test on real BEHAVE data. You can download them here:  [ProciGen](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.2VUEUS), [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/).

Once downloaded, modify `behave_dir` and `procigen_dir` in `configs/structured.py` to your local path.  

#### Pre-process data
We provide a split file including all image names from ProciGen and test image names from BEHAVE, [download here](https://edmond.mpg.de/file.xhtml?fileId=247856&version=2.1). 

To train stage 1 segmentation model, you will need to precompute the occupancy of human and object. We provide processed data for ProciGen [here](https://edmond.mpg.de/file.xhtml?fileId=247856&version=2.1). Download and unzip the file. 
Then you can run `python scripts/process_occ.py -o <unzipped occ files> -p <ProciGen path>` to process the downloaded occupancy file. 

Alternatively, you can process the sequence using: `python scripts/compute_occ.py -s <path to one sequence>`.

#### Train 
We train our stage 1 and stage 2 models separately in parallel to reduce training time. You can find example commands in `scripts/train.sh`. The checkpoint and logs will be saved to `outputs`. 

The data split file can be downloaded from [Edmond](https://edmond.mpg.de/file.xhtml?fileId=251365&version=4.0). 

## Evaluation
#### Pre-trained checkpoints
We provide checkpoints for models trained only on synthetic ProciGen, download them from [here](https://datasets.d2.mpi-inf.mpg.de/cvpr24procigen/pretrained.zip). 

Unzip the file and place them under folder `outputs`.

For more pretrained models, you can checkout our [Huggingface model card](https://huggingface.co/xiexh20/HDM-models/tree/main). 
Specifically, `stagex-<obj>` means the original stage x model first trained on full ProciGen data and then fine tuned on synthetic data of category `<obj>`.
`stagex-tune` means the model first trained on full ProciGen then fine tuned on training set of BEHAVE and InterCap. 

#### Run inference on BEHAVE test set
We provide example command to run batch inference at `scripts/test.sh`. Similar to training, the two stages are also run separately to reduce runtime. The example commands are 
configured to run inference on the pretrained checkpoints. 

Save as the training setup, the split file can be downloaded from [Edmond](https://edmond.mpg.de/file.xhtml?fileId=251365&version=4.0). 
#### Evaluate results 
After inference is done, results can be evaluated with: 
```shell
python eval/eval_separate.py -pr outputs/stage2/single/<save_name>/pred -gt outputs/stage2/single/<save_name>/gt -split configs/splits/behave-test.json 
```
The numbers will be saved to `./results`.

#### Results with DDIM inference
From June 12, 2024 on, we support DDIM at inference time, which is 10x faster. The performance remains similar, we report the results on BEHAVE and InterCap below (F-score@0.01m):

| BEHAVE | Human | Object  | Combined |
|:-------|:-----:|:-------:|:--------:|
| DDPM   |0.3477 | 0.4351  |  0.4110  |
| DDIM   |0.3533 | 0.4366  | 0.4170   |

| InterCap | Human | Object  | Combined |
|:---------|:-----:|:-------:|:--------:|
| DDPM     |0.3851 | 0.4928  |  0.4530  |
| DDIM     |0.3835 | 0.4817  | 0.4477   |


Note here the models are trained on synthetic ProciGen only, corresponding to *Ours synth. only* in our paper Table 2. We run DDIM for 100 steps in total with eta=1.0.  

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
