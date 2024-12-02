import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from decision_transformer.training.trainer import Trainer

class SequenceTrainer(Trainer):
    def __init__(self, lower_bound, **kwargs):
        super().__init__(**kwargs)
        if self.model.pskd:
            self.loss_fn = joint_loss(weight_state=0.05, weight_action=0.9, weight_rtg=0.05, pskd=self.model.pskd, lower_bound=lower_bound)

    def train_step(self, alpha_t):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            attention_mask,
        ) = self.get_batch(self.batch_size)
        
        action_target = torch.clone(actions)

        state_preds, action_preds, rtg_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            attention_mask=attention_mask,
        )

        self.step += 1
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        if self.model.predict_sr:
            state_target, rtg_target = torch.clone(states), torch.clone(rtg[:,:-1])

            state_preds = state_preds[:, :-1].reshape(-1, self.model.state_dim)[attention_mask[:, 1:].reshape(-1) > 0]
            state_target = state_target[:, 1:].reshape(-1, self.model.state_dim)[attention_mask[:, 1:].reshape(-1) > 0]

            rtg_preds = rtg_preds[:, :-1].reshape(-1, 1)[attention_mask[:, 1:].reshape(-1) > 0]
            rtg_target = rtg_target[:, 1:].reshape(-1, 1)[attention_mask[:, 1:].reshape(-1) > 0]
        else:
            state_target, state_preds = None, None
            rtg_target, rtg_preds = None, None

        if self.model.pskd:
            loss = self.loss_fn(
                state_preds, 
                action_preds, 
                rtg_preds,
                state_target, 
                action_target, 
                rtg_target,
                alpha_t,
            )
        else:
            loss = self.loss_fn(
                None,
                action_preds,
                None,
                None,
                action_target,
                None,
            )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics["training/action_error"] = (
                torch.mean((action_preds - action_target) ** 2).detach().cpu().item()
            )

        return loss.detach().cpu().item()#, lm_loss.detach().cpu().item()


class joint_loss:
    def __init__(self, weight_state=1, weight_action=1, weight_rtg=1, pskd=False, lower_bound=0.5):
        self.weight_state = weight_state
        self.weight_action = weight_action
        self.weight_rtg = weight_rtg

        self.huber_loss = torch.nn.SmoothL1Loss()
        self.mse_loss = torch.nn.MSELoss()

        self.pskd = pskd
        self.lower_bound = lower_bound

    def __call__(self, state_preds, action_preds, rtg_preds, state_target, action_target, rtg_target, alpha_t=None):
        if state_preds is not None:
            loss_s = self.huber_loss(state_preds, state_target)
            loss_r = self.huber_loss(rtg_preds, rtg_target)
        else:
            loss_s = torch.tensor(0).to(action_preds.device)
            loss_r = torch.tensor(0).to(action_preds.device)
            self.weight_action = 1

        if self.pskd:
            assert alpha_t is not None, "In the PSKD mode, `alpha_t` is not allowed to be None."
            alpha_t = min(self.lower_bound, alpha_t)
            alpha_t = 0
            soft_action_target = ((1 - alpha_t) * action_target) + (alpha_t * action_preds)
            loss_a = torch.mean((action_preds - soft_action_target) ** 2)
        else:
            loss_a = torch.mean((action_preds - action_target) ** 2)

        all_loss = self.weight_state * loss_s + self.weight_action * loss_a + self.weight_rtg * loss_r

        return all_loss
