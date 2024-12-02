export PYTHONPATH=./

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

model_type=dt # bc, dt
sample_ratio=1
lmlr=1e-5 # default is lr
dropout=0.1
warmup_steps=2500 # default is 10000
num_steps_per_iter=2500 # default is 2500
max_iters=40 # default is 40
num_eval_episodes=20 # default is 100

gpu=${1}
env=${2}
if [ "$env" = "reacher2d" ]; then
    K=5
else
    K=${10}
fi # K is context length
dataset=${3}
seed=${4}
pretrained_lm=${5}
lr=${6} # default is 1e-4
lora=${7}
lora_lr=${8}
weight_decay=${9} # default is 1e-4
predict_sr=${11}
mlp_embedding=${12}
pskd=${13}
alpha_T=${14}
lower_bound=${15}
group_lr=${16}
from_scratch=${17}

description="${env}_${dataset}_${pretrained_lm}_${lr}_${weight_decay}_predict-sr-${predict_sr}_pskd-${pskd}_alpha_T-${alpha_T}_lower-bound-${lower_bound}_group-lr-${group_lr}_seed-${seed}_K-${K}_mlp_embedding-${mlp_embedding}-${from_scratch}"
outdir="checkpoints/${description}"

CUDA_VISIBLE_DEVICES=${gpu} python experiment-d4rl/experiment.py --env ${env} \
        --dataset ${dataset} \
        --model_type ${model_type} \
        --seed ${seed} \
        --K ${K} \
        -lr ${lr} \
        -lmlr ${lmlr} \
        --num_steps_per_iter ${num_steps_per_iter} \
        --weight_decay ${weight_decay} \
        --max_iters ${max_iters} \
        --num_eval_episodes ${num_eval_episodes} \
        --sample_ratio ${sample_ratio} \
        --warmup_steps ${warmup_steps} \
        --pretrained_lm ${pretrained_lm} \
        --adapt_mode \
        --adapt_embed \
        --mlp_embedding ${mlp_embedding} \
        --outdir ${outdir} \
        --dropout ${dropout} \
        --description ${description} \
        --lora ${lora} \
        --predict_sr ${predict_sr} \
        --pskd ${pskd} \
        --alpha_T ${alpha_T} \
        --lower_bound ${lower_bound} \
        --group_learning_rate ${group_lr} \
        --from_scratch ${from_scratch} \
        --log_to_wandb 