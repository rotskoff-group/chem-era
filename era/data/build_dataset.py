from bpo.data.dataset_base import SMILESDataset, SMILESDatasetPrecompute
import torch
import numpy as np

def create_dataset(dataset_args: dict, dtype: torch.dtype, device: torch.device) -> SMILESDataset:
    return SMILESDataset(
        dtype = dtype, 
        device = device, 
        **dataset_args
    ), dataset_args