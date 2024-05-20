import era.models
import torch
import torch.nn as nn
from typing import Any


def create_model(model_args: dict, dtype: torch.dtype, device: torch.device) -> nn.Module:
    """Creates the model by passing argument dictoinaries into fetched constructors
    Args:
        model_args: The dictionary of model arguments, possibly a highly nested structure
        dtype: The datatype to use for the model
        device: The device to use for the model
    """
    model_base = getattr(era.models, model_args['model_type'])
    model_config = model_args['model_args']
    model = model_base(dtype=dtype,
                       device=device,
                       **model_config)
    if model_args['load_model'] is not None:
        print(f"Loading the following inside create_model: {model_args['load_model']}")
        ckpt = torch.load(model_args['load_model'], map_location='cpu')[
            "model_state_dict"]
        curr_state_dict = model.state_dict()
        pretrained_dict = {k: v
                           for k, v in ckpt.items()
                           if curr_state_dict[k].shape == v.shape}

        #Shows keys that are not going to be loaded
        print("Following keys in curr state dict but not in pretrained dict:")
        print(set(curr_state_dict.keys()) - set(pretrained_dict.keys()))
        print("Following keys are in pretrained dict but not in curr state dict:")
        print(set(pretrained_dict.keys()) - set(curr_state_dict.keys()))

        model.load_state_dict(pretrained_dict, strict=False)

    # Freeze requisite components
    model.freeze()

    # Update the model args
    model_args['model_args'] = model_config

    return model, model_args
