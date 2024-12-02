import numpy as np
import torch
import tqdm
import time
from itertools import cycle
import wandb


class Trainer:
    def __init__(
        self,
        args,
        model,
        optimizer,
        batch_size,
        get_batch,
        loss_fn,
        train_nlp_dataset=None,
        eval_nlp_dataset=None,
        scheduler=None,
        eval_fns=None,
        eval_only=False,
        log_to_wandb=False,
        max_iters=5,
    ):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.scaler = torch.cuda.amp.GradScaler()
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.step = 0
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.eval_only = eval_only
        self.eval_nlp_dataset = cycle(iter(eval_nlp_dataset)) if eval_nlp_dataset else None
        self.train_nlp_dataset = cycle(iter(train_nlp_dataset)) if train_nlp_dataset else None

        self.start_time = time.time()
        self.log_to_wandb = log_to_wandb
        self.max_iters = max_iters

        self.previous_max_norm_score = None
        self.previous_max_epoch = None

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        cur_step = (iter_num - 1) * num_steps
        train_losses = []
        # lm_losses = []
        logs = dict()

        train_start = time.time()

        if not self.eval_only:
            self.model.train()
            mean_loss = 0
            if self.model.pskd:
                alpha_t = self.model.alpha_T * ((iter_num) / self.max_iters)
                alpha_t = max(0, alpha_t)
            else:
                alpha_t = 0
            progress_bar = tqdm.tqdm(range(num_steps), desc=f"Training")
            for _ in progress_bar:
                # train_loss, lm_loss = self.train_step()
                train_loss = self.train_step(alpha_t)
                train_losses.append(train_loss)
                # lm_losses.append(lm_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

                cur_step += 1
                logs["time/training"] = time.time() - train_start
                logs["training/train_loss"] = train_loss
                logs["training/train_loss_mean"] = np.mean(train_losses)
                logs["training/train_loss_std"] = np.std(train_losses)
                # logs["training/lm_loss_mean"] = np.mean(lm_losses)
                # logs["training/lm_loss_std"] = np.std(lm_losses)
                logs["training/lr"] = self.scheduler._last_lr[1]
                logs["training/lmlr"] = self.scheduler._last_lr[0]

                progress_bar.set_postfix({"loss": logs["training/train_loss_mean"], "lr": self.optimizer.param_groups[0]['lr']})
                if self.log_to_wandb: wandb.log(logs, step=cur_step)


        eval_start = time.time()

        self.model.eval()
        eval_log = dict()
        for eval_fn in tqdm.tqdm(self.eval_fns, desc="Evaluating"):
            outputs = eval_fn(self.model)
            print(outputs)
            for k, v in outputs.items():
                print(k,":",v)
                logs[f"evaluation/{k}"] = v
                eval_log[f"evaluation/{k}"] = v

        if not self.eval_only:
            logs["time/total"] = time.time() - self.start_time
            eval_log["time/total"] = time.time() - self.start_time

        logs["time/evaluation"] = time.time() - eval_start
        if self.log_to_wandb: wandb.log(eval_log, step=cur_step)

        diagnostics_log = {}
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]
            diagnostics_log[k] = self.diagnostics[k]
        if self.log_to_wandb: wandb.log(diagnostics_log, step=cur_step)

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        max_norm_scores = [v for k, v in eval_log.items() if "normalized" in k]
        if not self.eval_only and 1 == 2:
            if self.args.get("outdir") and (self.previous_max_norm_score is None or \
                                            self.previous_max_norm_score < max(max_norm_scores)):
                self.previous_max_norm_score = round(max(max_norm_scores), 2)
                torch.save(
                    self.model.state_dict(),
                    f"{self.args['outdir']}/model_{iter_num}_{self.previous_max_norm_score}.pt",
                )

        return logs

    def train_step(self):
        self.optimizer.zero_grad()
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
            target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )

        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
