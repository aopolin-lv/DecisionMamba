import numpy as np
import torch
import imageio
import os
import cv2
from moviepy.editor import ImageSequenceClip

def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
    state_mean=0.0,
    state_std=1.0,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    scale=1000.0,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    target_return=None,
    mode="normal",
    record_video=False,
    video_path=None,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    frames = []

    if mode == "noise":
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        1, 1
    )
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != "delayed":
            pred_return = target_return[0, -1] - (reward / scale)
        else:
            pred_return = target_return[0, -1]
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)],
            dim=1,
        )

        episode_return += reward
        episode_length += 1
        
        if record_video and t % 2 == 0:
            curr_frame = env.render(mode='rgb_array')
            print(curr_frame.shape)
            frame_resized = cv2.resize(curr_frame, (84, 84)) 
            frames.append(frame_resized)

        if done:
            break

    model.past_key_values = None

    if record_video:
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path, logger=None)

    return episode_return, episode_length
