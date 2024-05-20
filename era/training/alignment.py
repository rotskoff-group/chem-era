import numpy as np
import pickle
import os
import shutil
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Optional
import re
from bpo.training.loss_fxns import BPOLoss


def alignment_loop(bpo: BPOLoss,
                   dataloader: torch.utils.data.DataLoader,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   writer: torch.utils.tensorboard.SummaryWriter,
                   write_freq: int = 100):
    tot_loss = 0
    bpo.policy.train()
    for ibatch, batch in enumerate(dataloader):
        inner_step = int(( epoch * len(dataloader)) + ibatch)
        optimizer.zero_grad()
        loss = bpo(batch)
        loss.backward()
        optimizer.step()
        if ibatch % write_freq == 0:
            print(f'Epoch {epoch}, batch {ibatch}, loss: {loss}')
        writer.add_scalar("Step KL Loss", loss.item(), inner_step)
    writer.add_scalar("Avg. Epoch KL Loss", tot_loss / len(dataloader), epoch)
    return tot_loss / len(dataloader)

def energy_distribution_loop(bpo: BPOLoss,
                             dataloader: torch.utils.data.DataLoader):
    all_energies = []
    bpo.policy.eval()
    for ibatch, batch in enumerate(dataloader):
        energies = bpo.get_energy_dist(batch)
        all_energies.append(energies.detach().cpu().numpy())
    return np.vstack(all_energies).flatten()

def align_policies(energy_model: list[nn.Module],
                   reference: nn.Module,
                   policy: nn.Module,
                   betas: list[float],
                   n_reps: int,
                   inference_options: dict,
                   nepochs: int,
                   ener_freq: int, 
                   ckpt_freq: int,
                   savedir: str,
                   importance_sample: bool,
                   regularize: bool,
                   prompted: bool,
                   gamma: float,
                   optimizer: torch.optim.Optimizer,
                   dataloader: torch.utils.data.DataLoader,
                   writer: torch.utils.tensorboard.SummaryWriter,
                   write_freq: int = 100,
                   dtype: torch.dtype = torch.float,
                   device: torch.device = None):

    bpo_framework = BPOLoss(energy_model, 
                            betas,
                            reference, 
                            policy,
                            n_reps, 
                            inference_options, 
                            importance_sample=importance_sample, 
                            regularize=regularize,
                            prompted=prompted,
                            gamma=gamma,
                            dtype=dtype, 
                            device=device)
    losses = []
    energies = []
    starting_energies = energy_distribution_loop(bpo_framework, dataloader)
    with open(f"{savedir}/energies_start.pkl", 'wb') as f:
        print("Saving energies before alignment")
        pickle.dump([starting_energies], f)
    for ep in range(nepochs):
        avg_loss = alignment_loop(bpo_framework, dataloader, optimizer, ep, writer, write_freq)
        losses.append(avg_loss)
        if ep % ener_freq == 0:
            print(f"Evaluating energy distribution on epoch {ep}")
            sub_eners = []
            energy = energy_distribution_loop(bpo_framework, dataloader)
            sub_eners.append(energy)
            energies.append(sub_eners)
            with open(f"{savedir}/energies_{ep}.pkl", 'wb') as handle:
                pickle.dump(sub_eners, handle)
        if ep % ckpt_freq == 0:
            print(f"Saving checkpoint on epoch {ep}")
            bpo_framework.save_policy(savedir=savedir, ep=ep)
    
    return energies, losses

