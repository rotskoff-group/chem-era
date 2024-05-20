import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable
from era.inference.inference_fxns import infer_SMILES_generator, prompted_generation
from functools import partial
import numpy as np
import h5py
from rdkit import Chem as chem
from torch.utils.data import Dataset

def save_generated_ERA_dataset(energies,
                               ref_logps, 
                               labels, 
                               masks,
                               savepath,
                               num_pred_per_prompt):
    with h5py.File(savepath, 'w') as f:
        f.create_dataset("energies", data=energies)
        f.create_dataset('ref_logps', data=ref_logps)
        f.create_dataset('tokens', data=labels)
        f.create_dataset('masks', data=masks)
        f.create_dataset('num_pred_per_prompt', data=num_pred_per_prompt)  

def identity_fn(labels, masks, pad_token):
    #Just returns the labels
    return labels

def separate_prompt_from_gen(labels, masks, pad_token):
    #Returns a tuplen of prompts and generations
    prompts = torch.ones_like(labels) * pad_token
    generations = torch.ones_like(labels) * pad_token
    for i in range(labels.shape[0]):
        curr_gen = labels[i, masks[i].bool()]
        curr_prompt = labels[i, ~masks[i].bool()] 
        generations[i][:len(curr_gen)] = curr_gen
        prompts[i][:len(curr_prompt)] = curr_prompt
    return (prompts, generations)

class ERAGenerator(nn.Module):

    def __init__(self,
                 energy_models: list[nn.Module],
                 energy_process_fxns: list[Callable],
                 reference: nn.Module,
                 inference_options: dict,
                 pad_token: int, 
                 prompted: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.ref = reference
        assert len(energy_models) == len(energy_process_fxns)
        self.energy_models = energy_models
        self.energy_fxns = [eval(x) for x in energy_process_fxns]
        for energy_model in self.energy_models:
            energy_model.to(self.dtype).to(self.device)
            energy_model.eval()
        self.inference_options = inference_options
        self.ref.to(self.dtype).to(self.device)
        self.ref.eval()
        self.pad_token = pad_token
        self.prompted = prompted

    def _sanitize_batch(self, batch: Tensor):
        x, _ = batch
        x = (x[0], None)
        y = (None, None)
        return (x, y)

    def _compute_logp(self, 
                      prob_logits: Tensor,
                      labels: Tensor,
                      masks: Tensor):
        """Computes the log probability of the indices given the probs"""
        logprobs = prob_logits.log_softmax(-1)
        labels = labels[:, 1:].clone()
        loss_mask = masks[:, 1:].clone()

        logprobs = torch.gather(logprobs[:, :-1, :],
                                dim=2,
                                index=labels.unsqueeze(2)).squeeze(2)
        logprobs = (logprobs * loss_mask).sum(-1)
        return logprobs
    
    def forward(self, batch: Tensor) -> Tensor:
        batch = self._sanitize_batch(batch)
        if self.prompted:
            print("prompted")
            labels, masks = prompted_generation(
                    self.ref, 
                    batch, 
                    opts = self.inference_options,
                    device = self.device
                )
        else:
            print("unprompted")
            labels = infer_SMILES_generator(self.ref,
                                            batch,
                                            opts = self.inference_options,
                                            device=self.device)
            masks = (labels != self.pad_token)
        labels = torch.tensor(labels).long().to(self.device)
        masks = torch.tensor(masks).long().to(self.device)
        #Generate the energies
        energies = torch.stack([energy((self.energy_fxns[i](labels, masks, self.pad_token), None)).detach() 
                                for i, energy in enumerate(self.energy_models)], dim=-1)
        assert labels.shape[0] == energies.shape[0]
        #Get the reference logps
        ref_probs_logits = self.ref((labels, None)).detach()
        ref_logps = self._compute_logp(ref_probs_logits, labels, masks)
        
        #Finally return everything 
        # import pdb; pdb.set_trace()
        energies = energies.detach().cpu().numpy()
        ref_logps = ref_logps.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        return energies, ref_logps, labels, masks

class ERADataset(Dataset):
    def __init__(self,
                 data_file: str,
                 load_to_memory: bool = False,
                 inference_mode: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        """
        Dataset for using precomputed alignment information
        """
        self.load_to_memory = load_to_memory
        self.device = device
        self.dtype = dtype
        self.data_file = data_file
        data_ptr = h5py.File(data_file, 'r')
        self.length = data_ptr['energies'].shape[0]
        self.num_pred_per_prompt = data_ptr['num_pred_per_prompt'][()]
        self.all_pairs = [[i, j] for i in range(self.num_pred_per_prompt)
                     for j in range(i + 1, self.num_pred_per_prompt)]
        del data_ptr
    
    def __len__(self):  
        return self.length
    
    def open_hdf5(self):
        self.data_hdf5 = h5py.File(self.data_file, 'r')
        self.energies = self.data_hdf5['energies']
        self.ref_logps = self.data_hdf5['ref_logps']
        self.tokens = self.data_hdf5['tokens']
        self.masks = self.data_hdf5['masks']
        self.data_opened = True

        if self.load_to_memory:
            self.energies = self.energies[()] 
            self.ref_logps = self.ref_logps[()] 
            self.tokens = self.tokens[()] 
            self.masks = self.masks[()] 
    
    def __getitem__(self, idx):
        if not hasattr(self, 'data_opened'):
            self.open_hdf5()
        
        prompt_idx = idx // self.num_pred_per_prompt
        pair_idx = idx % self.num_pred_per_prompt
        pair = self.all_pairs[pair_idx]
        idx1 = prompt_idx * self.num_pred_per_prompt + pair[0]
        idx2 = prompt_idx * self.num_pred_per_prompt + pair[1]
        assert idx1 != idx2
        return (
            self.tokens[idx1], self.tokens[idx2],
            self.masks[idx1], self.masks[idx2],
            self.energies[idx1], self.energies[idx2],
            self.ref_logps[idx1], self.ref_logps[idx2]
        )