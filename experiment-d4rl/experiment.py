import gym
import d4rl
import datetime
import numpy as np
import torch
import wandb
import argparse
import pickle
import random
import sys
import os
import loralib as lora
from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from get_nlp_datasets import get_dataset
from utils import get_optimizer, str_to_bool
from transformers import set_seed

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum

def experiment(
    exp_prefix,
    variant,
):
    torch.manual_seed(variant["seed"])
    set_seed(variant["seed"])
    os.makedirs(variant["outdir"], exist_ok=True)
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    if variant["co_training"]:
        print("co training with lambda="+str(variant["co_lambda"]))

    train_nlp_dataloader, eval_nlp_dataloader = None, None #get_dataset(
    #     dataset_name=variant["nlp_dataset_name"],
    #     dataset_config_name=variant["nlp_dataset_config_name"]
    # )

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    description = variant["description"]
    seed = variant["seed"]
    # group_name = f"{env_name}-{dataset}-{model_type}-{description}"
    group_name = f"train-epoch{variant['max_iters']}-iter{variant['num_steps_per_iter']}_eval-{variant['num_eval_episodes']}"
    timestamp = datetime.datetime.now()
    formatted_time = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    exp_prefix = f"{description}-{seed}-{formatted_time}"
    
    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    description = variant["description"]
    seed = variant["seed"]
    # group_name = f"{env_name}-{dataset}-{model_type}-{description}"
    group_name = f"train-epoch{variant['max_iters']}-iter{variant['num_steps_per_iter']}_eval-{variant['num_eval_episodes']}"
    timestamp = datetime.datetime.now()
    formatted_time = timestamp.strftime("%Y-%m-%d_%H:%M:%S")
    exp_prefix = f"{description}-{seed}-{formatted_time}"

    if env_name == "hopper":
        env = gym.make("hopper-medium-v2")
        max_ep_len = 1000
        env_targets = [3600]#, 2600, 2200, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
        variant["mlp_embedding"] = False
    elif env_name == "halfcheetah":
        env = gym.make("halfcheetah-medium-v2")
        max_ep_len = 1000
        if variant["dataset"] == "medium-expert":
            env_targets = [12000]
        else:
            env_targets = [12000, 8000, 6000, 4500]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("walker2d-medium-v2")
        max_ep_len = 1000
        env_targets = [5000]#, 4000, 3000, 2500]
        scale = 1000.0
    elif env_name == "ant":
        env = gym.make(f"ant-{variant['dataset']}-v2")
        max_ep_len = 1000
        env_targets = [3600, 6000]
        scale = 1000.0
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == "kitchen":
        env = gym.make("kitchen-complete-v0")
        max_ep_len = 1000
        env_targets = [5, 4, 3, 2, 1]
        scale = 1.0
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
        env_targets = env_targets[
            :1
        ]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    data_suffix = variant["data_suffix"]
    ratio_str = "-" + str(variant["sample_ratio"]) + "-" + data_suffix if variant["sample_ratio"] < 1 else ""
    if env_name in ["walker2d", "hopper", "halfcheetah", "reacher2d", "ant"]:
        dataset_path = f"data/mujoco/{env_name}-{dataset}{ratio_str}-v2.pkl"
    elif env_name == "kitchen":
        dataset_path = f"data/kitchen/{env_name}-{dataset}{ratio_str}-v0.pkl"
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
    num_eval_episodes = variant["num_eval_episodes"]
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

    def eval_episodes(target_rew, visualize):
        def fn(model):
            returns, lengths, video_paths = [], [], []
            os.makedirs(os.path.join(variant["outdir"], "videos", str(target_rew)), exist_ok=True)
            for episode_index in range(num_eval_episodes):
                record_video = (episode_index % (num_eval_episodes // 5) == 0) & visualize
                # if dir doesn't exist, make it
                if record_video:
                    video_path = os.path.join(
                        variant["outdir"], "videos", str(target_rew), f"episode_{episode_index}.mp4"
                    )
                    video_paths.append(video_path)
                else: 
                    video_path = None
                with torch.no_grad():
                    if model_type == "dt":
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            record_video = record_video,
                            video_path = video_path,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)

            return {
                f"target_{target_rew}_return_mean": np.mean(returns),
                f"target_{target_rew}_return_std": np.std(returns),
                f"target_{target_rew}_length_mean": np.mean(lengths),
                f"target_{target_rew}_length_std": np.std(lengths),
                f"target_{target_rew}_normalized_return_mean": env.get_normalized_score(np.mean(returns)),
                f"target_{target_rew}_videos": [wandb.Video(video_path, fps=30, format="mp4") for video_path in video_paths]
            }

        return fn
    
    if model_type == "dt":
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
            
        if variant["adapt_mode"]:
            if variant["lora"] == False:
                for param in model.parameters():
                    # param.requires_grad = False
                    param.requires_grad = True
            else:
                print("adapt lora.")
                lora.mark_only_lora_as_trainable(model, bias='lora_only')
                # lora.mark_only_lora_as_trainable(model, bias='all')
                # NOTE: Don't put this part below other adaptation part.
            if variant["adapt_wte"]:
                print("adapt wte.")
                for param in model.transformer.wte.parameters():
                    param.requires_grad = True
            if variant["adapt_wpe"]:
                print("adapt wpe.")
                for param in model.transformer.wpe.parameters():
                    param.requires_grad = True
            if variant["adapt_embed"]:
                print("adapt embeddings.")
                # adapt the embeddings in DecisionTransformer
                for name, param in model.named_parameters():
                    if ("embed" in name or "predict" in name):
                        param.requires_grad = True
            if variant["adapt_ln"]:
              print("adapt layer norms.")
              # adapt the LayerNorm in the transformer's blocks
              for block in model.transformer.h:
                  for param in block.ln_1.parameters():
                      param.requires_grad = True
                  for param in block.ln_2.parameters():
                      param.requires_grad = True
              # adapt the final LayerNorm in the transformer
              for param in model.transformer.ln_f.parameters():
                  param.requires_grad = True
            if variant["adapt_attn"]:
                print("adapt attention.")
                for block in model.transformer.h:
                # adapt the attention weights and biases
                    for param in block.attn.parameters():
                        param.requires_grad = True
            if variant["adapt_ff"]:
                print("adapt feed-forward.")
                for block in model.transformer.h:
                    # adapt the feed_forward weights and biases
                    for param in block.mlp.parameters():
                        param.requires_grad = True
            if variant["only_adapt_last_two_blocks"]:
                print("for transformer, only adapt the last two blocks.")
                for block in model.transformer.h[0:-2]:
                    for param in block.parameters():
                        param.requires_grad = False
            if variant["adapt_last_two_blocks"]:
                print("for transformer, adapt the last two blocks.")
                for block in model.transformer.h[-2:]:
                    for param in block.parameters():
                        param.requires_grad = True
        else: 
            print("fintune all.")
            
    elif model_type == "bc":
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
        )
    else:
        raise NotImplementedError

    trainable_param_size = 0
    frozen_param_size = 0
    for name, param in model.named_parameters():
        if "transformer" not in name: continue
        if param.requires_grad:
            trainable_param_size += param.numel()
        else:
            frozen_param_size += param.numel()
    print(f"Trainable parameters: {trainable_param_size}")
    print(f"Frozen parameters: {frozen_param_size}")
    print(f"Trainable ratio: {trainable_param_size/(trainable_param_size + frozen_param_size)}")
    
    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = get_optimizer(args=variant, model=model)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda steps: min((steps + 1) / warmup_steps, 1))
    
    visualize = variant["visualize"]

    if "dt" in model_type:
        trainer = SequenceTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            train_nlp_dataset=train_nlp_dataloader,
            eval_nlp_dataset=eval_nlp_dataloader,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar, visualize) for tar in env_targets],
            log_to_wandb=log_to_wandb,
            max_iters=variant["max_iters"],
            lower_bound=variant["lower_bound"],
        )
    elif model_type == "bc":
        trainer = ActTrainer(
            args=variant,
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            train_nlp_dataset=train_nlp_dataloader,
            eval_nlp_dataset=eval_nlp_dataloader,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar, visualize) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            # NOTE: fill in the name of your own wandb project
            # entity="reproduce",
            project="decision_mamba_from_scratch",
            config=variant,
        )
        # wandb.watch(model)  # wandb has some bug

    total_training_time = 0
    for iter in range(variant["max_iters"]):
        print("HI!")
        outputs = trainer.train_iteration(
            num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        print("HI2!")
        total_training_time += outputs["time/training"]
        outputs["time/total_training_time"] = total_training_time
        if log_to_wandb:
            # wandb.log(outputs)
            wandb.log({"time/training": total_training_time}, step=(iter + 1) * variant["num_steps_per_iter"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="medium"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
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
    # lm co-training
    parser.add_argument("--co_training", action="store_true", default=False)
    parser.add_argument("--nlp_dataset_name", type=str, default="wikitext")
    parser.add_argument(
        "--nlp_dataset_config_name", type=str, default="wikitext-103-raw-v1"
    )
    parser.add_argument("--co_lambda", type=float, default=0.1)

    parser.add_argument("--predict_sr", type=str_to_bool, default=False)
    parser.add_argument("--pskd", type=str_to_bool, default=False)
    parser.add_argument("--alpha_T", type=float, default=1.0)
    parser.add_argument("--lower_bound", type=float, default=0.5)
    parser.add_argument("--group_learning_rate", type=float, default=2e-5)
    parser.add_argument("--from_scratch", type=str_to_bool, default=False)

    args = parser.parse_args()
    experiment("d4rl-experiment", variant=vars(args))
