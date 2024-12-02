import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

import sys
import os
# use ../../decision_transformer as decision_transformer when run as main
if __name__=="__main__":
    sys.path.insert(0, os.path.abspath('../..'))
    sys.path.insert(0, os.path.abspath('..'))

from decision_transformer.models.model import TrajectoryModel
from transformers.models.gpt2 import GPT2Tokenizer
from transformers import MambaForCausalLM, MambaConfig
from decision_transformer.models.trajectory_gpt2 import GPT2Model, GPT2LMHeadModel
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Model_LoRA, GPT2LMHeadModel_LoRA
from decision_transformer.models.trajectory_gpt2_LoRA import GPT2Config_LoRA, MambaConfig_LoRA

from decision_transformer.models.utils import ResidualBlock, MLPBlock
from peft import LoraConfig, get_peft_model

MODEL_HUB = {
    "mamba-130m": "/home/lv_qi/DecisionMamba/mamba_config",
    "mamba-370m": "ArthurZ/mamba-370m",
    "mamba-790m": "ArthurZ/mamba-790m",
    "mamba-1.4b": "ArthurZ/mamba-1.4b",
    "mamba-2.8b": "ArthurZ/mamba-2.8b",
}

class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """
    @property
    def transformer(self):
        return self.transformer_model.backbone
      
    def __init__(
        self,
        args,
        state_dim,
        act_dim,
        hidden_size,
        max_length=None,
        max_ep_len=4096,
        action_tanh=True,
        predict_sr=False,
        pskd=False,
        alpha_T=None,
        **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.pskd = pskd
        self.predict_sr = predict_sr
        self.hidden_size = hidden_size
        self.alpha_T = alpha_T

        mamba_path = MODEL_HUB.get(args["pretrained_lm"], "mamba-130m")

        if args["pretrained_lm"] is not None and not args["from_scratch"]:
            print("Loading from pretrained "+args["pretrained_lm"]+" model")
            if args['lora']:
                # config = GPT2Config_LoRA.from_pretrained(args["pretrained_lm"], lora_attn_dim=16)
                # self.transformer_model = GPT2LMHeadModel_LoRA.from_pretrained(
                #     args["pretrained_lm"],
                #     config=config
                # )
                config = MambaConfig_LoRA.from_pretrained(mamba_path, lora_attn_dim=16)
                self.transformer_model = MambaForCausalLM.from_pretrained(mamba_path, config=config)

                for mambablock in self.transformer_model.backbone.layers:
                    self.copy_params(mambablock.mixer, mambablock.mixer_group)

                # modules_to_save = ["embed_tokens"]
                target_modules = ["x_proj", "dt_proj", "in_proj", "out_proj"]
                lora_config = LoraConfig(
                    r=16, lora_alpha=32, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
                    # modules_to_save=modules_to_save
                )
                self.lora_config = lora_config
                self.transformer_model = get_peft_model(self.transformer_model, lora_config)
                print("lora enable")
            else:
                # config = transformers.GPT2Config.from_pretrained(args["pretrained_lm"])
                # config.resid_pdrop = args["dropout"]
                # self.transformer_model = GPT2LMHeadModel.from_pretrained(
                #     args["pretrained_lm"],
                #     config=config,
                # )
                config = MambaConfig.from_pretrained(mamba_path)
                config.resid_pdrop = args["dropout"]
                self.transformer_model = MambaForCausalLM.from_pretrained(mamba_path)
            # hidden_size = config.n_embd
            # self.hidden_size = config.n_embd
            hidden_size = config.hidden_size
            self.hidden_size = config.hidden_size

        else:
            
            if args['lora']:
                config = GPT2Config_LoRA.from_pretrained("gpt2")
                self.transformer_model = GPT2LMHeadModel_LoRA(config)
            else:
                config = MambaConfig.from_pretrained(mamba_path)
                config.resid_pdrop = args["dropout"]
                self.transformer_model = MambaForCausalLM(config=config)
            hidden_size = config.hidden_size
            self.hidden_size = config.hidden_size

        if max_ep_len > 1024 and args["extend_positions"]:
            current_max_pos, embed_size = self.transformer.wpe.weight.shape
            new_encoder_pos_embed = self.transformer.wpe.weight.new_empty(
                max_ep_len, embed_size
            )
            # copy position embeddings over and over to initialize the new position embeddings
            orig_k = 2
            k = orig_k
            step = current_max_pos - k
            new_encoder_pos_embed[:k] = self.transformer.wpe.weight[:k]
            while k < max_ep_len - 1:
                new_encoder_pos_embed[k : (k + step)] = self.transformer.wpe.weight[
                    orig_k : min(max_ep_len - k + orig_k, current_max_pos)
                ]
                k += step
            self.transformer.wpe.weight.data = new_encoder_pos_embed

        if args["extend_positions"]:
            self.embed_timestep = self.transformer.wpe
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
    
        if args["mlp_embedding"]:
            self.embed_return = ResidualBlock(1, hidden_size)
            self.embed_state = ResidualBlock(self.state_dim, hidden_size)
            self.embed_action = ResidualBlock(self.act_dim, hidden_size)
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        if args["mlp_embedding"]:
          if args["share_input_output_proj"]: raise ValueError("With MLP in embeddings, you cannot share the projections")
          self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
          self.predict_action = MLPBlock(self.hidden_size, self.act_dim, self.hidden_size)
          self.predict_rtg = torch.nn.Linear(hidden_size, 1)
        else:
          if args["share_input_output_proj"]:
            self.predict_state = lambda x: F.linear(x, self.embed_state.weight.t())
            self.predict_rtg = lambda x: F.linear(x, self.embed_return.weight.t())
            self.predict_action = lambda x: F.tanh(
                F.linear(x, self.embed_action.weight.t())
            )
          else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
            self.predict_action = nn.Sequential(
                *(
                    [nn.Linear(hidden_size, self.act_dim)]
                    + ([nn.Tanh()] if action_tanh else [])
                )
            )
            self.predict_rtg = torch.nn.Linear(hidden_size, 1)
        
        self.past_key_values = None

        # set group params require_grad to true
        for name, param in self.named_parameters():
            if "mixer_group" in name:
                param.requires_grad = True
        
        print(self)

    def copy_params(self, model_from, model_to):
        params_from = model_from.named_parameters()
        params_to = model_to.named_parameters()

        dict_params_to = dict(params_to)
        for name, param in params_from:
            if name in dict_params_to:
                dict_params_to[name].data.copy_(param.data)

    def forward(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        attention_mask=None,
        past_key_values=None,
    ):

        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)
        # embed each modality with a different head
        
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
       
        stacked_inputs = (
            torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            )
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_length, self.hidden_size)
        )
        all_embs = self.embed_ln(stacked_inputs)

        stacked_inputs = all_embs + time_embeddings.repeat_interleave(3, dim=1)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = (
            torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(batch_size, 3 * seq_length)
        )

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
            past_key_values=None,  # self.past_key_values,
            use_cache=True,
        )
        x = transformer_outputs["last_hidden_state"]
        # self.past_key_values = transformer_outputs["past_key_values"]

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        action_preds = self.predict_action(x[:, 1])  # predict next action given state

        # ======= add state/return-to-go prediction network
        if self.predict_sr:
            state_preds = self.predict_state(x[:, 2])
            rtg_preds = self.predict_rtg(x[:, 2])
        else:
            state_preds = None
            rtg_preds = None
        # ======= add state/return-to-go prediction network

        return state_preds, action_preds, rtg_preds, transformer_outputs["last_hidden_state"]

    def get_action(
        self,
        states,
        actions,
        rewards,
        returns_to_go,
        timesteps,
        past_key_values=None,
        **kwargs
    ):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length :]
            actions = actions[:, -self.max_length :]
            returns_to_go = returns_to_go[:, -self.max_length :]
            timesteps = timesteps[:, -self.max_length :]

            # pad all tokens to sequence length
            attention_mask = torch.cat(
                [
                    torch.zeros(self.max_length - states.shape[1]),
                    torch.ones(states.shape[1]),
                ]
            )
            attention_mask = attention_mask.to(
                dtype=torch.long, device=states.device
            ).reshape(1, -1)
            states = torch.cat(
                [
                    torch.zeros(
                        (
                            states.shape[0],
                            self.max_length - states.shape[1],
                            self.state_dim,
                        ),
                        device=states.device,
                    ),
                    states,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            actions = torch.cat(
                [
                    torch.zeros(
                        (
                            actions.shape[0],
                            self.max_length - actions.shape[1],
                            self.act_dim,
                        ),
                        device=actions.device,
                    ),
                    actions,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [
                    torch.zeros(
                        (
                            returns_to_go.shape[0],
                            self.max_length - returns_to_go.shape[1],
                            1,
                        ),
                        device=returns_to_go.device,
                    ),
                    returns_to_go,
                ],
                dim=1,
            ).to(dtype=torch.float32)
            timesteps = torch.cat(
                [
                    torch.zeros(
                        (timesteps.shape[0], self.max_length - timesteps.shape[1]),
                        device=timesteps.device,
                    ),
                    timesteps,
                ],
                dim=1,
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds, _ = self.forward(
            states,
            actions,
            None,
            returns_to_go,
            timesteps,
            attention_mask=attention_mask,
            **kwargs,
        )

        return action_preds[0, -1]