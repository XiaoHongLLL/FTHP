#!/bin/bash

## model parameter
device=1 # GPU device number
data=../data/data_mimic/fold1/ # datasets directory
batch=1 # batch size
n_head=3 # number of heads of transformer encoder
n_layers=3 # number of layers of transformer encoder
d_model=64 # dimension of the transformer model
d_rnn=64 # dimension of RNN layer (not used now)
d_inner_hid=256 # dimension of the hidden layer in transformer model (Feed Forward Network)
d_k=16
d_v=16
optimizer=adam # optimizer
scheduler=cosLR # type of lr scheduler; 'steplr' for StepLR, 'reduce' for ReduceLROnPlateau
dropout=0.1
lr=1e-4
epoch=40
eval_epoch=50 # number of epoch we want to start to evaluate
normalize=normal
loss_lambda=1.0 # 辅助类型损失的权重

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
save_path=./checkpoints/mimic/
save_name=mimic_fthp_model_best_${normalize}_lambda_${loss_lambda}

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
load_path_name=./checkpoints/mimic/${save_name}.pth
save_result=./results/mimic/${save_name}_samples

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