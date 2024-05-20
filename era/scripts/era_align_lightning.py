import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from era.models import create_model
from copy import deepcopy
import hydra
from era.scripts.top_level_utils import (
    seed_worker,
    dtype_convert,
    split_data_subsets,
    specific_update
)
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from era.models.alignment_module import AlignmentModule
from omegaconf import OmegaConf
from lightning.pytorch.strategies import FSDPStrategy
from era.data.alignment_dataset import ERADataset
import torch
import pickle
import os

@hydra.main(version_base="1.3", config_path="../cfgs", config_name="alignment")
def main(cfg) -> None:
    global_args = cfg['global_args']
    alignment_args = cfg['alignment']
    energy_model_args = cfg['energy_model']
    policy_model_args = cfg['policy_model']
    reference_model_args = cfg['reference_model']
    data_args = cfg['data']

    L.seed_everything(global_args['seed'])

    #Set up the policy model only
    policy_metadata = pickle.load(open(policy_model_args['model_metadata'], 'rb'))
    policy_model_args = specific_update(policy_model_args, policy_metadata)
    alignment_args = specific_update(alignment_args, policy_metadata)

    #Add some energy models as well in case we need to evaluate energy distributions
    #   during alignment
    energy_keys = [f'model_{i}' for i in range(len(energy_model_args) - 1)] #Subtract 1 for model_metadata key
    energy_models = []
    for k in energy_keys:
        inner_args = energy_model_args[k]
        if hasattr(inner_args, 'ensemble_paths') and inner_args['ensemble_paths'] is not None:
            ensemble_paths = inner_args['ensemble_paths']
            for path in ensemble_paths:
                print(f"Loading from {path}")
                tmp_energy_args = deepcopy(inner_args)
                tmp_energy_args['load_model'] = path
                energy_model, _ = create_model(tmp_energy_args, dtype = torch.float, device=None)
                energy_models.append(energy_model)
        else:
            energy_model, _ = create_model(inner_args, dtype=torch.float, device=None)
            energy_models.append(energy_model)

    #Set up dataset
    dataset = ERADataset(**data_args)
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(dataset, 
                            worker_init_fn=seed_worker, 
                            generator=g,
                            **alignment_args['dloader_args'])

    #Model checkpoints
    every_epoch_checkpoint_callback = ModelCheckpoint(filename="model_{epoch:02d}",
                                                          every_n_epochs=alignment_args['ckpt_freq'],
                                                          save_top_k=-1)
    
    best_checkpoint_callback = ModelCheckpoint(filename="best_model",
                                                   monitor='validation/total_loss',
                                                   mode="min",
                                                   save_top_k=1)
    logger = TensorBoardLogger(save_dir="./")
    alignment_model = AlignmentModule(policy_model_args,
                                      alignment_args,
                                      energy_models)
    
    if (global_args['ngpus'] != 0):
        if alignment_args['strategy'] == 'fsdp':
            strategy = FSDPStrategy(
                activation_checkpointing_policy={nn.TransformerEncoderLayer},
                sharding_strategy="FULL_SHARD"
            )
        else:
            strategy = alignment_args['strategy']
        
        trainer = L.Trainer(
            max_epochs = alignment_args['nepochs'],
            logger = logger,
            callbacks = [every_epoch_checkpoint_callback, best_checkpoint_callback],
            accelerator='cuda',
            devices=global_args['ngpus'],
            strategy=strategy,
            inference_mode=False,
            precision="16-mixed"
        )
    else:
        trainer = L.Trainer(
            max_epochs = alignment_args['nepochs'],
            logger = logger,
            callbacks = [every_epoch_checkpoint_callback, best_checkpoint_callback],
            accelerator = 'cpu',
            inference_mode=False,
            precision="16-mixed"
        )
    
    if trainer.global_rank == 0:
        train_folder_name = trainer.logger.log_dir
        os.makedirs(train_folder_name, exist_ok=True)
        tot_config = {
            'global_args' : global_args,
            'alignment' : alignment_args,
            'policy_model' : policy_model_args,
            'energy_model' : energy_model_args,
            'reference_model' : reference_model_args,
            'data' : data_args
        }
        OmegaConf.save(tot_config, f"{train_folder_name}/config.yaml")
    
    if alignment_args['eval_energy_only']:
        print("Only evaluating trained policy model!")
        trainer.test(alignment_model, dataloader)
    else:
        print("Performing alignment")
        trainer.fit(alignment_model, dataloader, dataloader,
                    ckpt_path = alignment_args['restart_checkpoint'] if alignment_args['restart_checkpoint'] is not None else None)
    