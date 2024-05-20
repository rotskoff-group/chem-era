import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from era.data import create_dataset
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
from era.models.lightning_module import LightningModel
from omegaconf import OmegaConf
from lightning.pytorch.strategies import FSDPStrategy
import torch
import os

@hydra.main(version_base="1.3", config_path="../cfgs", config_name="training")
def main(cfg) -> None:
    global_args = cfg['global_args']
    dataset_args = cfg['data']
    model_args = cfg['model']
    training_args = cfg['training']
    dtype, device = dtype_convert(global_args['dtype']), None #Device handled by ptl trainer

    dataset, updated_dataset_args = create_dataset(dataset_args, dtype, device)
    size_dict = dataset.get_sizes()
    token_dict = dataset.get_ctrl_tokens()
    max_len_dict = dataset.get_max_seq_len()
    total_dict = {**size_dict, **token_dict, **max_len_dict}
    # Fix target pad token as ignore index
    tgt_pad_token = total_dict['tgt_pad_token']
    total_dict['ignore_index'] = tgt_pad_token if tgt_pad_token is not None else -100
    total_dict['seed'] = global_args['seed']

    print("Setting up dataloaders...")
    train_set, val_set, test_set = split_data_subsets(dataset,
                                                      training_args['splits'],
                                                      training_args['train_size'],
                                                      training_args['val_size'],
                                                      training_args['test_size'])
    
    g = torch.Generator()
    g.manual_seed(0)

    if training_args['sample']:
        #TODO: Compute weight based on train set, add to dataloader
        weights = train_set.dataset.get_weights(train_set.indices)
        sampler = WeightedRandomSampler(weights, len(train_set), replacement=True, generator=g)
        #Pass sampler into dataloader?
    else:
        sampler = None

    train_loader = DataLoader(train_set, worker_init_fn=seed_worker, generator=g, sampler=sampler,
                              **training_args['dloader_args'])
    val_loader = DataLoader(val_set, worker_init_fn=seed_worker,
                            generator=g, **training_args['dloader_args'])
    test_loader = DataLoader(test_set, worker_init_fn=seed_worker,
                             generator=g, **training_args['dloader_args'])

    # Update model args
    model_args = specific_update(model_args, total_dict)
    # Update training args
    training_args = specific_update(training_args, total_dict)

    L.seed_everything(global_args['seed'])
    savedir = global_args['savedir']

    every_epoch_checkpoint_callback = ModelCheckpoint(filename="model_{epoch:02d}_loss={validation_loss:.2f}",
                                                          every_n_epochs=1,
                                                          save_top_k=-1)
    best_checkpoint_callback = ModelCheckpoint(filename="best_model",
                                                   monitor='validation_loss',
                                                   mode="min",
                                                   save_top_k=1)
    logger = TensorBoardLogger(save_dir = "./")
    lightning_model = LightningModel(
        model_args = model_args,
        training_args = training_args
    )

    if (global_args['ngpus'] != 0):
        if training_args['strategy'] == 'fsdp':
            strategy = FSDPStrategy(
                activation_checkpointing_policy={nn.TransformerEncoderLayer},
                sharding_strategy="FULL_SHARD"
            )
        else:
            strategy = training_args['strategy']
        trainer = L.Trainer(
            max_epochs = training_args['nepochs'],
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
            max_epochs = training_args['nepochs'],
            logger = logger,
            callbacks = [every_epoch_checkpoint_callback, best_checkpoint_callback],
            accelerator = 'cpu',
            precision="16-mixed"
        )
    
    if trainer.global_rank == 0:
        train_folder_name = trainer.logger.log_dir
        os.makedirs(train_folder_name, exist_ok=True)
        tot_config = {
            'global_args' : global_args,
            'data' : updated_dataset_args,
            'model' : model_args,
            'training' : training_args
        }
        OmegaConf.save(tot_config, f"{train_folder_name}/config.yaml")
    
    print(f"Reloading from {training_args['restart_checkpoint']}")
    trainer.fit(lightning_model, train_loader, val_loader,
                ckpt_path = training_args['restart_checkpoint'] if training_args['restart_checkpoint'] is not None else None)



    



