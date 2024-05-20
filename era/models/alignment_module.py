import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import era.models as models
import era.training.loss_fxns as loss_fxns
from era.inference.inference_fxns import prompted_generation, infer_SMILES_generator
from era.data.alignment_dataset import identity_fn, separate_prompt_from_gen
import pickle
import os

class AlignmentModule(L.LightningModule):
    def __init__(self,
                 model_args: dict,
                 alignment_args: dict,
                 energy_models: dict):
        super().__init__()
        self.model_args = model_args
        self.energy_models = energy_models
        self.alignment_args = alignment_args
        self.build_model_base()
        if self.model_args['load_model'] is not None:
            self.load_model_from_checkpoint(self.model_args['load_model'])
        self.save_hyperparameters(ignore='energy_models')

    def build_model_base(self):
        model_base = getattr(models, self.model_args['model_type'])
        model_config = self.model_args['model_args']
        self.model = model_base(**model_config)

    def load_model_from_checkpoint(self, filename):
        print(f"Loading model from ckpt file {filename}")
        ckpt = torch.load(filename)['state_dict']
        #Remove the model. prefix from all checkpoint keys
        ckpt = {'.'.join(k.split('.')[1:]) : v for k, v in ckpt.items()}
        curr_state_dict = self.model.state_dict()
        pretrained_dict = {k : v
                           for k, v in ckpt.items()
                           if curr_state_dict[k].shape == v.shape}
        
        print("Following keys in curr state dict but not in pretrained dict:")
        print(set(curr_state_dict.keys()) - set(pretrained_dict.keys()))
        print("Following keys are in pretrained dict but not in curr state dict:")
        print(set(pretrained_dict.keys()) - set(curr_state_dict.keys()))
        
        self.model.load_state_dict(pretrained_dict, strict=False)

    def configure_optimizers(self):
        optimizer_base = getattr(optim, self.alignment_args['optimizer'])
        optimizer = optimizer_base(self.model.parameters(), **self.alignment_args['optimizer_args'])
        return optimizer
    
    def _compute_logp(self, 
                      prob_logits,
                      labels,
                      masks):
        """Computes the log probability of the indices given the probs"""
        logprobs = prob_logits.log_softmax(-1)
        labels = labels[:, 1:].clone()
        loss_mask = masks[:, 1:].clone()

        logprobs = torch.gather(logprobs[:, :-1, :],
                                dim=2,
                                index=labels.unsqueeze(2)).squeeze(2)
        logprobs = (logprobs * loss_mask).sum(-1)
        return logprobs

    def _shared_eval(self, batch, batch_idx, prefix):
        tokens_1, tokens_2, masks_1, masks_2, energies_1, energies_2, ref_logps1, ref_logps2 = batch

        #Define beta' and gamma'
        beta_prime = torch.tensor(self.alignment_args['betas']).to(energies_1.device)
        beta_prime = beta_prime / (1 + self.alignment_args['gamma'])
        gamma_prime = self.alignment_args['gamma'] / (1 + self.alignment_args['gamma'])

        #These should broadcast correctly
        #Energy shape: (batch_size, 1, num_energies)
        energies_1 = energies_1 * beta_prime
        energies_2 = energies_2 * beta_prime
        #(batch_size,)
        energies_1 = energies_1.sum((-1, -2))
        energies_2 = energies_2.sum((-1, -2))
        policy_probs_logits_1 = self.model((tokens_1, None))
        policy_probs_logits_2 = self.model((tokens_2, None))
        #Compute the log probabilities
        #(batch_size,)
        policy_logprobs_1 = self._compute_logp(policy_probs_logits_1, tokens_1, masks_1)
        policy_logprobs_2 = self._compute_logp(policy_probs_logits_2, tokens_2, masks_2)

        logp = nn.functional.logsigmoid(policy_logprobs_2 - policy_logprobs_1)
        logp_prime = nn.functional.logsigmoid(policy_logprobs_1 - policy_logprobs_2)

        #Reformulation of the KL loss
        logp_star = nn.functional.logsigmoid(-(energies_2 - energies_1) + (gamma_prime * (ref_logps2 - ref_logps1)))
        logp_star_prime = nn.functional.logsigmoid(-(energies_1 - energies_2) + (gamma_prime * (ref_logps1 - ref_logps2)))

        kl_loss = torch.exp(logp) * (logp - logp_star) + torch.exp(logp_prime) * (logp_prime - logp_star_prime)

        if self.alignment_args['regularize']:
            pass
            #(batch_size,)
            ref_sum = ref_logps2 + ref_logps1
            policy_sum = policy_logprobs_2 + policy_logprobs_1
            reg_loss = (ref_sum - policy_sum)
        else:
            pass
            reg_loss = 0
        
        tot_loss = kl_loss
        
        tot_loss = torch.mean(tot_loss)
        metrics = {f"{prefix}/total_loss" : tot_loss}
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)
        return tot_loss
        
    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")
    
    def _sanitize_batch_on_policy_sample_prompted(self, batch):
        tokens_1, tokens_2, masks_1, masks_2, energies_1, energies_2, ref_logps1, ref_logps2 = batch
        prompts = tokens_1 * (1-masks_1)
        prompts[prompts == 0] = 324
        return ((prompts, None), None)
    
    def _sanitize_batch_on_policy_sample_unprompted(self, batch):
        tokens_1, tokens_2, masks_1, masks_2, energies_1, energies_2, ref_logps1, ref_logps2 = batch
        #We only need the first token which should be a start token
        #   (hard code the check for now)
        tokens_1 = tokens_1[:, 0].unsqueeze(-1)
        return ((tokens_1, None), None)
    
    def _eval_save_energy_distribution(self, dataloader, num_batches_for_energy):
        raise NotImplementedError
        #Only evaluate the energy distribution once per test step
        if os.path.exists(f"{self.logger.log_dir}/energy_distribution_{self.current_epoch}.pkl"):
            return
        print("Getting energy distribution ")
        all_energies = []
        for idx, batch in enumerate(dataloader):
            if idx >= num_batches_for_energy:
                break
            with torch.enable_grad():
                if self.alignment_args['prompted']:
                    batch_proc = self._sanitize_batch_on_policy_sample_prompted(batch)
                    labels, masks = prompted_generation(
                        self.model, 
                        batch_proc, 
                        opts = self.alignment_args['inference_options'],
                        device = self.device
                        )
                else:
                    batch_proc = self._sanitize_batch_on_policy_sample_unprompted(batch)
                    labels = infer_SMILES_generator(self.model,
                                                           batch_proc, 
                                                           opts = self.alignment_args['inference_options'],
                                                           device=self.device
                                                           )
                    masks = (labels != 324)
                labels = torch.tensor(labels).long().to(self.device)
                masks = torch.tensor(masks).long().to(self.device)
                betas = torch.tensor(self.alignment_args['betas']).to(self.device)
                appl_fn = separate_prompt_from_gen if self.alignment_args['prompted'] else identity_fn
                energies = torch.stack([betas[i] * energy((appl_fn(labels, masks, 324), None)).to(self.device).detach() 
                                    for i, energy in enumerate(self.energy_models)], dim=-1).sum(-1)
                energies = energies.flatten()
                all_energies.append(energies)
        all_energies = torch.cat(all_energies)
        #Save the energies
        savedir = self.logger.log_dir
        with open(f"{savedir}/energy_distribution_{self.current_epoch}.pkl", 'wb') as f:
            pickle.dump(all_energies, f)

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            return self._shared_eval(batch, batch_idx, "validation")
    
    def on_save_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:
        #Also save an energy distribution for the model but only if 
        #requested (mostly for energy evaluation purposes)
        if (self.trainer.local_rank == 0) and (self.alignment_args['eval_energy_dists']):
            self._eval_save_energy_distribution(self.trainer.val_dataloaders, self.alignment_args['num_batches_for_energy_dist'])
        return super().on_save_checkpoint(checkpoint)

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            self._eval_save_energy_distribution(self.trainer.test_dataloaders, self.alignment_args['num_batches_for_energy_dist'])
            return self._shared_eval(batch, batch_idx, "test")
