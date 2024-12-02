from torch import optim
from itertools import chain
# import timm.optim.optim_factory as optim_factory
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS


def get_optimizer(args, model):
    if args["pretrained_lm"]:

        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        group_layers = [n for n, p in model.named_parameters() if "group" in n and p.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in group_layers and n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": args["weight_decay"],
                "lr": args["group_learning_rate"],
                "eps":1e-6,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n in group_layers and n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": args["group_learning_rate"],
                "eps":1e-6,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in group_layers and n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": args["weight_decay"],
                "lr": args["learning_rate"],
                "eps":1e-6,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if (n not in group_layers and n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": args["learning_rate"],
                "eps":1e-6,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters)
    else:
        for name, param in model.named_parameters():
            print(f"{name}: {param.requires_grad}")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args["learning_rate"],
            weight_decay=args["weight_decay"],
        )  
    return optimizer


def str_to_bool(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        raise ValueError("Input value must be 'True' or 'False'")