#!/bin/bash

# Example command to run test on BEHAVE dataset
# Tip: You can speed up inference by appending `run.diffusion_scheduler=ddim run.num_inference_steps=100` to the end of the command,
# same for both stage 1 and stage 2 inference.

# Stage 1: reconstruct H+O and segment, results will be saved to outputs/run.name/single/sample
python main.py \
run.name=stage1 model.consistent_center=True \
model.image_feature_model=vit_base_patch16_224_mae dataloader.batch_size=16 \
model.model_name=pc2-diff-ho-sepsegm model.predict_binary=True model.lw_binary=3.0 \
dataset=behave dataset.max_points=16384 \
scheduler=linear optimizer.lr=3e-4 \
dataset.split_file=your_split_file run.job=sample


# Stage 2: load stage 1 results and refine human and object
python main.py \
run.name=stage2 model.consistent_center=True \
model.image_feature_model=vit_base_patch16_224_mae dataloader.batch_size=16 \
model=ho-attn model.attn_weight=1.0 model.attn_type=coord3d+posenc-learnable \
dataset=behave dataset.type=behave-attn model.point_visible_test=combine \
dataset.split_file=your_split_file run.job=sample \
run.save_name=stage1-500step run.sample_noise_step=500 run.sample_mode=interm-pred \
dataset.ho_segm_pred_path=$PWD/outputs/stage1/single/sample/pred # path to the first stage predictions