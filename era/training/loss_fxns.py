import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from bpo.inference.inference_fxns import infer_SMILES_generator, prompted_generation
from functools import partial
import numpy as np
from rdkit import Chem as chem

CrossEntropyLoss = nn.CrossEntropyLoss
BCELoss = nn.BCELoss
BCELossWithLogits = nn.BCEWithLogitsLoss
MSELoss = nn.MSELoss


class BPOLoss(nn.Module):

    def __init__(self,
                 energy_models: list[nn.Module],
                 betas: list[float],
                 reference: nn.Module,
                 policy: nn.Module,
                 n_reps: int,
                 inference_options: dict,
                 importance_sample: bool = False,
                 regularize: bool = False,
                 prompted: bool = False,
                 gamma: float = 1.0,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.importance_sample = importance_sample
        self.betas = betas
        self.regularize = regularize
        self.gamma = gamma
        self.classifiers = energy_models
        self.ref = reference
        self.policy = policy
        #Energy model and reference are both in eval mode
        for classifier in self.classifiers:
            classifier.to(self.dtype).to(self.device)
            classifier.eval()
        self.ref.to(self.dtype).to(self.device)
        self.ref.eval()
        #Policy is in train mode
        self.policy.to(self.dtype).to(self.device)
        self.policy.train()
        self.n_reps = n_reps
        assert(self.n_reps == 2)
        self.inference_opts = inference_options
        alphabet = np.load(self.inference_opts['alphabet'], allow_pickle=True)  
        self.alphabet = np.array(alphabet)
        self.generator_type = self.inference_opts['model_type']
        self.pad_token = len(alphabet)
        assert(self.pad_token == 324)
        self.prompted = prompted
        #Do that at the script level
        # self.classifier.network.output_head.network.1 = nn.functional.logsigmoid()

    def _sanitize_batch(self, batch: Tensor):
        x, _ = batch
        if self.prompted:
            x = (torch.repeat_interleave(x[0], self.n_reps, dim=0).to(self.device), None)
        else:
            x = (torch.repeat_interleave(x[0], self.n_reps, dim=0).reshape(-1, 1).to(self.device), None)
        y = (None, None)
        return (x, y)

    def forward(self, batch: Tensor):
        # Can recycle algorithm from inference_fxns
        batch = self._sanitize_batch(batch)
        if self.prompted:
            labels = prompted_generation(
                self.ref, 
                batch, 
                opts = self.inference_opts,
                device = self.device
            )
        elif not self.prompted:
            labels = infer_SMILES_generator(self.ref,
                                            batch,
                                            opts=self.inference_opts,
                                            device=self.device)
        labels = torch.tensor(labels).long().to(self.device) #(n_smi, max_len)
        #Add the prompts onto the labels, will use them later
        if self.prompted:
            prompts = batch[0][0]
            labels = (prompts, labels)
        
        energies = torch.stack([self.betas[i] * -nn.functional.logsigmoid(classifier((labels, None)).detach()) 
                                for i, classifier in enumerate(self.classifiers)], dim=-1).sum(-1)
        energies = energies.reshape(-1, self.n_reps)
        
        if self.prompted:
            #Convert labels back
            labels = labels[1]
            assert(325 in labels)
            assert(326 in labels)

        #These are generator functions
        print(labels.shape)
        policy_probs_logits = self.policy((labels, None))
        policy_logps = self._compute_logp(policy_probs_logits, labels).reshape(-1, self.n_reps)

        logp = nn.functional.logsigmoid(policy_logps[:,1] - policy_logps[:,0])
        logp_prime = nn.functional.logsigmoid(policy_logps[:,0] - policy_logps[:,1])
        logp_star = nn.functional.logsigmoid(-(energies[:,1] - energies[:,0]))
        logp_star_prime = nn.functional.logsigmoid(-(energies[:,0] - energies[:,1]))
        kl_loss = torch.exp(logp) * (logp - logp_star) + torch.exp(logp_prime) * (logp_prime - logp_star_prime) 

        if self.regularize or self.importance_sample:
            ref_probs_logits = self.ref((labels, None)).detach()
            ref_logps = self._compute_logp(ref_probs_logits, labels).reshape(-1, self.n_reps)
        else:
            ref_probs_logits = None
            ref_logps = None

        if self.regularize:
            reg_loss = self.gamma * (ref_logps.sum(-1) - policy_logps.sum(-1))
            kl_loss = kl_loss + reg_loss

        if self.importance_sample:
            raise NotImplementedError
            #Conduct importance sampling against the reference
            importance_weight = self._compute_importance_weight(policy_logps, ref_logps)
            kl_loss = kl_loss * importance_weight

        value = torch.mean(kl_loss)
        try:
            assert not torch.isnan(value)
        except:
            raise ValueError("NaN detected!")
        return value

    def _compute_logp(self, 
                      prob_logits: Tensor,
                      labels: Tensor):
        """Computes the log probability of the indices given the probs"""
        logprobs = prob_logits.log_softmax(-1)
        labels = labels[:, 1:].clone()
        loss_mask = (labels != self.pad_token)

        logprobs = torch.gather(logprobs[:, :-1, :],
                                dim=2,
                                index=labels.unsqueeze(2)).squeeze(2)
        logprobs = (logprobs * loss_mask).sum(-1)
        return logprobs
    
    def _compute_importance_weight(self, 
                                   policy_logps: Tensor,
                                   ref_logps: Tensor) -> Tensor:
        """Computes the importance weight for the policy"""
        return torch.exp(policy_logps.sum(-1) - ref_logps.sum(-1))

    def get_energy_dist(self, batch: Tensor):
        batch = self._sanitize_batch(batch)
        if self.prompted:
            labels = prompted_generation(
                self.policy, 
                batch, 
                opts = self.inference_opts,
                device = self.device
            )
        else:
            labels = infer_SMILES_generator(self.policy,
                                            batch,
                                            opts = self.inference_opts,
                                            device=self.device)

        labels = torch.tensor(labels).long().to(self.device)
        if self.prompted:
            prompts = batch[0][0]
            labels = (prompts, labels)
        energies = torch.stack([self.betas[i] * -nn.functional.logsigmoid(classifier((labels, None)).detach()) 
                                for i, classifier in enumerate(self.classifiers)], dim=-1).sum(-1)
        return energies
    
    def save_policy(self, savedir: str, ep: int) -> None:
        """Saves the policy model after alignment"""
        torch.save({
            'model_state_dict' : self.policy.state_dict(),
            'epoch' : ep
        }, f"{savedir}/policy_model_{ep}.pt")
