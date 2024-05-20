import torch
import torch.nn as nn
import torch.optim as optim

def create_optimizer(model: nn.Module, 
                     model_args: dict, 
                     training_args: dict, 
                     dtype: torch.dtype, 
                     device: torch.device) -> torch.optim.Optimizer:
    
    optimizer_raw = getattr(optim, training_args['optimizer'])
    optimizer = optimizer_raw(model.parameters(), **training_args['optimizer_args'])
    if (model_args['load_model'] is not None) and (model_args['load_optimizer']):
        ckpt = torch.load(model_args['load_model'], map_location=device)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return optimizer
