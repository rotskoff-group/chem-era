import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from era.models import create_model
from era.training import create_optimizer
from era.training.alignment import align_policies
import hydra
from era.scripts.top_level_utils import (
    seed_everything,
    seed_worker,
    dtype_convert,
    save_completed_config,
    specific_update
)
from copy import deepcopy
from era.data import create_dataset

class ContextDatasetDummy(Dataset):

    def __init__(self, start_token: int, length: int = 128):
        self.data = torch.tensor([start_token])
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx: int): 
        return (self.data[0], 0), (0, 0)

@hydra.main(version_base="1.3", config_path="../cfgs", config_name="alignment")
def main(cfg) -> None:
    global_args = cfg['global_args']
    alignment_args = cfg['alignment']
    energy_model_args = cfg['energy_model']
    policy_model_args = cfg['policy_model']
    reference_model_args = cfg['reference_model']

    energy_keys = [f'model_{i}' for i in range(len(energy_model_args) - 1)] #Subtract 1 for model_metadata key
    print(energy_keys)

    print("Parsing arguments...")
    print("Setting up device, datatype, and seed...")
    device = torch.device('cuda:0' if global_args['ngpus'] > 0 else 'cpu')
    dtype = dtype_convert(global_args['dtype'])
    seed = seed_everything(global_args['seed'])
    print("Running on device", device, "with datatype", dtype, "and seed", seed)

    # Construct each model and update the model args
    energy_metadata = pickle.load(open(energy_model_args['model_metadata'], 'rb'))
    energy_model_args = specific_update(energy_model_args, energy_metadata)

    policy_metadata = pickle.load(open(policy_model_args['model_metadata'], 'rb'))
    policy_model_args = specific_update(policy_model_args, policy_metadata)

    reference_metadata = pickle.load(
        open(reference_model_args['model_metadata'], 'rb'))
    reference_model_args = specific_update(
        reference_model_args, reference_metadata)

    alignment_args = specific_update(alignment_args, policy_metadata)

    # Create the models now, taking into account multiple replicas
    energy_models = []
    for k in energy_keys:
        inner_args = energy_model_args[k]
        if hasattr(inner_args, 'ensemble_paths') and inner_args['ensemble_paths'] is not None:
            ensemble_paths = inner_args['ensemble_paths']
            for path in ensemble_paths:
                print(f"Loading from {path}")
                tmp_energy_args = deepcopy(inner_args)
                tmp_energy_args['load_model'] = path
                energy_model, _ = create_model(tmp_energy_args, dtype, device)
                energy_models.append(energy_model)
        else:
            energy_model, _ = create_model(inner_args, dtype, device)
            energy_models.append(energy_model)

    policy_model, updated_policy_args = create_model(policy_model_args, dtype, device)
    reference_model, _ = create_model(reference_model_args, dtype, device)

    print(energy_models)
    print(f"Using {len(energy_models)} energy model components")
    print(policy_model)
    print(reference_model)

    #Set up optimizer and bind it ONLY to the policy model
    optimizer = create_optimizer(policy_model, updated_policy_args, alignment_args, dtype, device)

    #Set up the data to load into the era with a dataloader, use dummy data of only start tokens for now
    if not alignment_args['prompted']:
        assert(policy_metadata['src_start_token'] == reference_metadata['src_start_token'])
        dataset = ContextDatasetDummy(policy_metadata['src_start_token'],
                                    alignment_args['dloader_args']['batch_size'])
    else:
        data_args = cfg['data']
        dataset, _ = create_dataset(data_args, dtype, device)
    
    # import pdb; pdb.set_trace()
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(dataset, worker_init_fn=seed_worker, 
                            generator=g,
                            sampler=None,
                            **alignment_args['dloader_args'])
    
    writer = SummaryWriter(log_dir=global_args['savedir'])

    tot_config = {
        'global_args': global_args,
        'alignment': alignment_args,
        'energy_model': energy_model_args,
        'policy_model': policy_model_args,
        'reference_model': reference_model_args
    }
    save_completed_config("full_alignment_config.yaml", tot_config, global_args['savedir'])
    print("Beginning alignment")

    energies, losses = align_policies(
        energy_models,
        reference_model, 
        policy_model,
        betas=alignment_args['betas'],
        n_reps=alignment_args['n_reps'],
        inference_options=alignment_args['inference_options'],
        nepochs=alignment_args['nepochs'],
        ener_freq=alignment_args['ener_freq'],
        ckpt_freq=alignment_args['ckpt_freq'],
        savedir=global_args['savedir'],
        importance_sample=alignment_args['importance_sample'],
        regularize=alignment_args['regularize'],
        prompted=alignment_args['prompted'],
        gamma=alignment_args['gamma'],
        optimizer=optimizer,
        dataloader=dataloader,
        writer=writer,
        dtype=dtype,
        device=device)
    with open(f"{global_args['savedir']}/energies.pkl", 'wb') as handle:
        pickle.dump(energies, handle)

    with open(f"{global_args['savedir']}/losses.pkl", 'wb') as handle:
        pickle.dump(losses, handle)
