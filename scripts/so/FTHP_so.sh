#!/bin/bash

## model parameter
device=1
data=../data/data_so/fold1/
batch=4
n_head=4
n_layers=4
d_model=512
d_rnn=64
d_inner_hid=1024 # 修正: d_inner -> d_inner_hid
d_k=512
d_v=512
optimizer=adam
scheduler=reduce
dropout=0.1
lr=1e-4
epoch=60
eval_epoch=61
normalize=None
loss_lambda=5 # 新增: 辅助损失权重 (参考 smurf_so.sh)

## (新) Flow Matching & GMM 参数
d_latent=16 # GMM 潜空间维度
fm_sigma=0.1 # Flow Matching 路径 sigma
solver_method=euler # ODE 求解器方法
solver_step_size=0.01 # ODE 求解器步长

## (新) sampling/evaluation 参数
n_samples=100 # number of generated sample for evaluation

## Quantile evaluation
eval_quantile=-1
eval_quantile_step=0.05

## model saving
save_path=./checkpoints/so/
save_name=so_fthp_model_best_${normalize}_lambda_${loss_lambda}

cd ../..

# --- 训练命令 ---
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py \
-data $data \
-normalize $normalize \
-loss_lambda $loss_lambda \
-batch $batch \
-n_head $n_head -n_layers $n_layers \
-d_model $d_model -d_rnn $d_rnn -d_inner_hid $d_inner_hid \
-d_k $d_k -d_v $d_v -dropout $dropout \
-lr $lr -epoch $epoch -eval_epoch $eval_epoch \
-optimizer $optimizer -scheduler $scheduler \
-d_latent $d_latent -fm_sigma $fm_sigma \
-solver_method $solver_method -solver_step_size $solver_step_size \
-n_samples $n_samples \
-eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step \
-save_name $save_name -save_path $save_path

# --- 评估命令 ---
load_path_name=./checkpoints/so/${save_name}.pth
save_result=./results/so/${save_name}_samples

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python main.py \
-data $data \
-normalize $normalize \
-loss_lambda $loss_lambda \
-just_eval \
-batch $batch \
-n_head $n_head -n_layers $n_layers \
-d_model $d_model -d_rnn $d_rnn -d_inner_hid $d_inner_hid \
-d_k $d_k -d_v $d_v -dropout $dropout \
-lr $lr -epoch $epoch -eval_epoch $eval_epoch \
-optimizer $optimizer -scheduler $scheduler \
-d_latent $d_latent -fm_sigma $fm_sigma \
-solver_method $solver_method -solver_step_size $solver_step_size \
-n_samples $n_samples \
-eval_quantile $eval_quantile -eval_quantile_step $eval_quantile_step \
-load_path_name $load_path_name -save_result $save_result