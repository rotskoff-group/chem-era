import torch
import numpy as np
import pickle
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
from era.data.alignment_dataset import ERAGenerator, save_generated_ERA_dataset
from copy import deepcopy
from era.data import create_dataset

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

    reference_model, _ = create_model(reference_model_args, dtype, device)

    print(energy_models)
    print(f"Using {len(energy_models)} energy model components")
    print(reference_model)

    #Set up the data to load into the bpo with a dataloader, use dummy data of only start tokens for now
    data_args = cfg['data']
    dataset, _ = create_dataset(data_args, dtype, device)
    
    # import pdb; pdb.set_trace()
    g = torch.Generator()
    g.manual_seed(0)
    dataloader = DataLoader(dataset, worker_init_fn=seed_worker, 
                            generator=g,
                            sampler=None,
                            **alignment_args['dloader_args'])

    tot_config = {
        'global_args': global_args,
        'alignment': alignment_args,
        'energy_model': energy_model_args,
        'policy_model': policy_model_args,
        'reference_model': reference_model_args
    }
    save_completed_config("full_alignment_config.yaml", tot_config, global_args['savedir'])
    print("Beginning generation")

    data_generator = ERAGenerator(energy_models,
                                  alignment_args['energy_process_fxns'],
                                  reference_model,
                                  alignment_args['inference_options'],
                                  alignment_args['pad_token'],
                                  alignment_args['prompted'],
                                  dtype=dtype,
                                  device=device)
    
    all_energies = []
    all_reflogps = []
    all_labels = []
    all_masks = []
    #One pass through the dataset
    for i, batch in enumerate(dataloader):
        # if i > 2:
        #     break
        energies, ref_logps, labels, masks = data_generator(batch)
        all_energies.append(energies)
        all_reflogps.append(ref_logps)
        all_labels.append(labels)
        all_masks.append(masks)
    #Find the maximum length for padding
    pad_token = alignment_args['pad_token']
    max_len = max([x.shape[-1] for x in all_labels])
    all_labels = np.concatenate([np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), 'constant', constant_values=pad_token) for x in all_labels])
    all_masks = np.concatenate([np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), 'constant', constant_values=0) for x in all_masks])
    assert all_labels.shape == all_masks.shape
    all_energies = np.concatenate(all_energies)
    all_reflogps = np.concatenate(all_reflogps)

    save_generated_ERA_dataset(all_energies,
                               all_reflogps,
                               all_labels,
                               all_masks,
                               f"{global_args['savedir']}/alignment_dataset.h5",
                               alignment_args['inference_options']['num_pred_per_tgt'])
