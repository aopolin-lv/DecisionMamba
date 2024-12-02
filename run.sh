#!/bin/sh

env=${1}
dataset=${2}
pretrained_lm=mamba-130m

K=20
seed=2
lr=1e-4
weight_decay=1e-5

lora=False
lora_lr=1e-3
group_lr=${lr}
mlp_embedding=True

gpu_index=${3}

predict_sr=True
pskd=True
alpha_T=${4}
lower_bound=0.5
from_scratch=True

nohup sh scripts/run_gym.sh ${gpu_index} ${env} ${dataset} \
    ${seed} ${pretrained_lm} ${lr} ${lora} ${lora_lr} ${weight_decay}  \
    ${K} ${predict_sr} ${mlp_embedding} ${pskd} ${alpha_T} ${lower_bound} ${group_lr} ${from_scratch}\
    > logs/${env}_${dataset}_${pretrained_lm}_lr-${lr}_wd-${weight_decay}_predict-sr-${predict_sr}_pskd-${pskd}_alpha_T-${alpha_T}_lower-bound-${lower_bound}_group-lr-${group_lr}_seed-${seed}_lora-${lora}-${lora_lr}_K-${K}_mlp-embed-${mlp_embedding}-${from_scratch}.log 2>&1 &
