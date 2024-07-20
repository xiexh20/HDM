#!/bin/bash

# Example commands to train HDM model

# Train stage 1 model with 4 GPUs:
python -m torch.distributed.run --nproc_per_node 4 main.py \
run.name=stage1-new model.consistent_center=True \
model.image_feature_model=vit_base_patch16_224_mae dataloader.batch_size=16 \
model.model_name=pc2-diff-ho-sepsegm model.predict_binary=True model.lw_binary=3.0 \
dataset=behave dataset.max_points=16384 \
scheduler=linear optimizer.lr=3e-4 \
dataset.split_file=your_split_file \
run.max_steps=500000


# To train stage 2 model with 4 GPUs:
python -m torch.distributed.run --nproc_per_node 4 main.py \
run.name=stage2-new model.consistent_center=True \
model.image_feature_model=vit_base_patch16_224_mae dataloader.batch_size=16 \
model=ho-attn model.attn_weight=1.0 model.attn_type=coord3d+posenc-learnable \
dataset=behave dataset.type=behave-attn model.point_visible_test=combine \
dataset.split_file=your_split_file \
run.max_steps=500000