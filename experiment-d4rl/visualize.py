"""可视化mamba矩阵"""
import torch

from decision_transformer.models.decision_transformer import DecisionTransformer
import numpy as np
import random
from experiment import discount_cumsum
import os
from transformers import set_seed
import gym
import math
import d4rl
import pickle
import argparse
from torch import nn
from utils import str_to_bool

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def add_hook(model: nn.Module):
    tassms = []
    for layer in model.transformer_model.backbone.layers:
        _tassms = []
        # for blk in (layer.mixers, layer.mixer_group):
        tassm = layer.mixer
        setattr(tassm, "__DEBUG__", True)
        _tassms.append(tassm)
        tassms.append(_tassms)
    return model, tassms

def add_hook_group(model: nn.Module):
    tassms = []
    for layer in model.transformer_model.backbone.layers:
        _tassms = []
        # for blk in (layer.mixers, layer.mixer_group):
        tassm = layer.mixer_group
        setattr(tassm, "__DEBUG__", True)
        _tassms.append(tassm)
        tassms.append(_tassms)
    return model, tassms

@torch.no_grad()
def main(variant):
    torch.manual_seed(variant["seed"])
    set_seed(variant["seed"])
    os.makedirs(variant["outdir"], exist_ok=True)
    device = variant.get("device", "cuda")

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    seed = variant["seed"]
    
    if env_name == "hopper":
        env = gym.make("hopper-medium-v2")
        max_ep_len = 1000
        env_targets = [3600]#, 2600, 2200, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
        variant["mlp_embedding"] = False
    elif env_name == "halfcheetah":
        env = gym.make("halfcheetah-medium-v2")
        max_ep_len = 1000
        env_targets = [12000, 8000, 6000, 4500]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("walker2d-medium-v2")
        max_ep_len = 1000
        env_targets = [5000, 4000, 3000, 2500]
        scale = 1000.0
    elif env_name == "ant":
        env = gym.make(f"ant-{variant['dataset']}-v2")
        max_ep_len = 1000
        env_targets = [3600, 6000]
        scale = 1000.0
    elif env_name == "antmaze":
        env = gym.make(f"{env_name}-{dataset}-v2")
        max_ep_len = 1000
        if dataset == "umaze-diverse":
            env_targets = [20]
        elif dataset == "umaze":
            env_targets = [5]
        scale = 1.0
    else:
        raise NotImplementedError

    if model_type == "bc":
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    data_suffix = variant["data_suffix"]
    ratio_str = "-" + str(variant["sample_ratio"]) + "-" + data_suffix if variant["sample_ratio"] < 1 else ""
    if env_name in ["walker2d", "hopper", "halfcheetah", "reacher2d", "ant"]:
        dataset_path = f"data/mujoco/{env_name}-{dataset}{ratio_str}-v2.pkl"
    else: 
        raise NotImplementedError
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    
    # save all path information into separate lists
    mode = variant.get("mode", "normal")
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == "delayed":  # delayed: all rewards moved to end of trajectory
            path["rewards"][-1] = path["rewards"].sum()
            path["rewards"][:-1] = 0.0
        states.append(path["observations"])
        traj_lens.append(len(path["observations"]))
        returns.append(-path["rewards"].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    variant["state_mean"], variant["state_std"] = state_mean, state_std

    num_timesteps = sum(traj_lens)

    print("=" * 50)
    print(f"Starting new experiment: {env_name} {dataset}")
    print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
    print(f"Average return: {np.mean(-returns):.2f}, std: {np.std(returns):.2f}")
    print(f"Max return: {np.max(-returns):.2f}, min: {np.min(-returns):.2f}")
    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]
    pct_traj = variant.get("pct_traj", 1.0)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)               # 得到从小到大排的index顺序
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]          # 按照轨迹的步长从小->大排，加入训练集
    ind = len(trajectories) - 2                     # 原本数据的index个数为len(trajectories)-1，这里上一步已经加入了一个，因此还要再减1
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories 按照轨迹长短sample，长轨迹分配更大的采样概率
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj["rewards"].shape[0] - 1)

            # get sequences from dataset
            s.append(traj["observations"][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            if "terminals" in traj:
                d.append(traj["terminals"][si : si + max_len].reshape(1, -1))
            else:
                d.append(traj["dones"][si : si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                discount_cumsum(traj["rewards"][si:], gamma=1.0)[
                    : s[-1].shape[1] + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -10.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        float_dtype = torch.float32
        
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=float_dtype, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=float_dtype, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=float_dtype, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=float_dtype, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask
    
    model = DecisionTransformer(
        args=variant,
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant["embed_dim"],
        n_layer=variant["n_layer"],
        n_head=variant["n_head"],
        n_inner=4 * variant["embed_dim"],
        activation_function=variant["activation_function"],
        n_positions=1024,
        resid_pdrop=variant["dropout"],
        attn_pdrop=0.1,
        mlp_embedding=variant["mlp_embedding"],
        fp16=variant["fp16"],
        predict_sr=True if variant["predict_sr"] else False,
        pskd=variant["pskd"],
        alpha_T=variant["alpha_T"],
    )

    model_path = "/home/lv_qi/project/DecisionMamba/checkpoints/hopper_medium_mamba-130m_1e-4_1e-5_predict-sr-True_pskd-True_alpha_T-0.85_lower-bound-0.5_group-lr-1e-4_seed-2_K-60_mlp_embedding-True-True/model_11_0.79.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    model = model.to(device=device)

    model, tassms = add_hook(model)
    model, tassms_local = add_hook_group(model)
    (
        states,
        actions,
        rewards,
        dones,
        rtg,
        timesteps,
        attention_mask,
    ) = get_batch(batch_size=batch_size)

    action_target = torch.clone(actions)

    state_preds, action_preds, rtg_preds, hidden_state = model.forward(
        states,
        actions,
        rewards,
        rtg[:, :-1],
        timesteps,
        attention_mask=attention_mask,
    )

    # ==== global ====
    regs = getattr(tassms[-2][-1], "__data__")
    As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
    us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
    print(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
    B, N, L = Bs.shape
    D, N = As.shape
    H, W = int(math.sqrt(L)), int(math.sqrt(L))

    dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, D, L)
    dw_logs = As.view(D, N)[None, :, :, None] * dts[:,:,None,:] # (B, D, N, L)
    ws = torch.cumsum(dw_logs, dim=-1).exp()

    Qs, Ks = Cs[:,None,:,:], Bs[:,None,:,:]
    _Qs, _Ks = Qs.view(-1, N, L), Ks.view(-1, N, L)
    attns = (_Qs.transpose(1, 2) @ _Ks).view(B, -1, L, L)
    attns = attns.mean(dim=1).detach().cpu().numpy()

    # ==== local ====
    regs = getattr(tassms_local[-2][-1], "__data__")
    As, Bs, Cs, Ds = -torch.exp(regs["A_logs"].to(torch.float32)), regs["Bs"], regs["Cs"], regs["Ds"]
    us, dts, delta_bias = regs["us"], regs["dts"], regs["delta_bias"]
    print(As.shape, Bs.shape, Cs.shape, Ds.shape, us.shape, dts.shape, delta_bias.shape)
    B, N, L = Bs.shape
    D, N = As.shape
    H, W = int(math.sqrt(L)), int(math.sqrt(L))

    dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, D, L)
    dw_logs = As.view(D, N)[None, :, :, None] * dts[:,:,None,:] # (B, D, N, L)
    ws = torch.cumsum(dw_logs, dim=-1).exp()

    Qs, Ks = Cs[:,None,:,:], Bs[:,None,:,:]
    _Qs, _Ks = Qs.view(-1, N, L), Ks.view(-1, N, L)
    attns_local = (_Qs.transpose(1, 2) @ _Ks).view(B, -1, L, L).mean(dim=1).detach().cpu()
    temp = attns_local.unsqueeze(2).unsqueeze(4)
    temp = temp.repeat(1,1,3,1,3)
    temp = temp.view(-1, K*3, K*3)
    attns_local = temp.numpy()

    # ==== combine ====
    attns_all = attns + attns_local

    att_hidden = hidden_state @ hidden_state.transpose(-1, -2)
    for i in range(0, batch_size):
        draw_all(attns, attns_local, att_hidden.detach().cpu().numpy(), i, variant["K"])


def draw_all(attns_global, attns_local, atts_all, index, k):
    # draw(attns_global, index, f"global_{index}", k)
    # draw(attns_local, index, f"local_{index}", k)
    draw(atts_all, index, f"all_{index}", k)


def draw(attns, index, prefix, k):
    plt.clf()
    data = attns[index]
    # data[np.tril_indices_from(data, k=-1)] = 0
    mean = np.mean(data)
    std = np.std(data)

    # 标准化数据
    normalized_data = (data - mean) / std

    cmap = "Blues"
    plt.imshow(normalized_data, cmap=cmap, interpolation='nearest', vmax=2.2, vmin=-4.2)
    plt.colorbar()  # 添加颜色条
    plt.show()
    if not os.path.exists(f"visualize_result/{k}"): os.makedirs(f"visualize_result/{k}")
    plt.savefig(f"visualize_result/{k}/{cmap}_{prefix}.png", dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument("--dataset", type=str, default="medium")  # medium, medium-replay, medium-expert, expert
    parser.add_argument("--mode", type=str, default="normal")  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_type", type=str, default="dt")  # dt for decision transformer, bc for behavior cloning
    # data sampling
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--data_suffix", type=str, default="d1")
    # training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)
    parser.add_argument("--visualize", "-v", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--description", type=str, default="")
    # architecture, don't need to care about in our method
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--extend_positions", action="store_true", default=False)
    parser.add_argument("--share_input_output_proj", action="store_true", default=False)
    # learning hyperparameters
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--lm_learning_rate", "-lmlr", type=float, default=None)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=40)
    parser.add_argument("--num_steps_per_iter", type=int, default=2500)
    # implementations
    parser.add_argument("--pretrained_lm", type=str, default=None)
    parser.add_argument("--mlp_embedding", type=str_to_bool, default=False)
    # adaptations
    parser.add_argument("--adapt_mode", action="store_true", default=False)
    parser.add_argument("--lora", type=str_to_bool, default=False)
    parser.add_argument("--only_adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_last_two_blocks", action="store_true", default=False)
    parser.add_argument("--adapt_ln", action="store_true", default=False)
    parser.add_argument("--adapt_attn", action="store_true", default=False)
    parser.add_argument("--adapt_ff", action="store_true", default=False)
    parser.add_argument("--adapt_embed", action="store_true", default=False)
    parser.add_argument("--adapt_wte", action="store_true", default=False)
    parser.add_argument("--adapt_wpe", action="store_true", default=False)

    parser.add_argument("--predict_sr", type=str_to_bool, default=False)
    parser.add_argument("--pskd", type=str_to_bool, default=False)
    parser.add_argument("--alpha_T", type=float, default=1.0)
    parser.add_argument("--lower_bound", type=float, default=0.5)
    parser.add_argument("--group_learning_rate", type=float, default=2e-5)
    parser.add_argument("--from_scratch", type=str_to_bool, default=True)

    args = parser.parse_args()
    main(vars(args))