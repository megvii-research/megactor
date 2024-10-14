import json
import torch
import torch.nn as nn


def match_name_keywords(n: str, name_keywords: list):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def get_param_dict(args, model_without_ddp: nn.Module):
    try:
        param_dict_type = args.param_dict_type
    except:
        param_dict_type = 'default'
    assert param_dict_type in ['default', 'ddetr_in_mmdet', 'large_wd']

    # by default
    if param_dict_type == 'default':
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and "bert" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model_without_ddp.named_parameters() if "bert" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]

        param_name_dicts = [
            {"params": [n for n, p in model_without_ddp.named_parameters() if "backbone" not in n and "bert" not in n and p.requires_grad]},
            {
                "params": [n for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [n for n, p in model_without_ddp.named_parameters() if "bert" in n and p.requires_grad],
                "lr": args.lr_backbone,
            }
        ]

        print('param_name_dicts: ', json.dumps(param_name_dicts, indent=2))

        return param_dicts, param_name_dicts




    raise NotImplementedError



        # print("param_dicts: {}".format(param_dicts))

    return param_dicts, None