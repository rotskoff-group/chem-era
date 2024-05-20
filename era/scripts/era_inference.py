import torch
from torch.utils.data import DataLoader
import pickle #Temporary
import argparse
import yaml
from era.data import create_dataset
from era.models import create_model
from era.inference.inference import run_inference
from .top_level_utils import (
    seed_everything,
    seed_worker,
    dtype_convert,
    split_data_subsets,
    save_inference_predictions,
    save_token_size_dict,
    specific_update,
    select_model,
    save_completed_config,
    divide_parallel_subsets
)
from typing import Any
import hydra

#Necessary functions for distributed data parallel inference

def get_args() -> dict:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description='Run NMR inference')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    parser.add_argument('local_rank', type = int, help = 'The local rank of this process')
    parser.add_argument('n_procs', type = int, help = 'The total number of concurrent processes running')
    args = parser.parse_args()
    listdoc =  yaml.safe_load(open(args.config_file, 'r'))
    return (
        listdoc['global_args'],
        listdoc['data'],
        listdoc['model'],
        listdoc['inference']
    ), (args.local_rank, args.n_procs)

@hydra.main(version_base="1.3", config_path="../cfgs", config_name="inference")
def main(cfg) -> None:
    print("Parsing arguments...")
    global_args = cfg['global_args']
    dataset_args = cfg['data']
    model_args = cfg['model']
    inference_args = cfg['inference']
    print("Setting up device, datatype, and seed...")
    local_rank, n_procs = inference_args['local_rank'], inference_args['n_procs']

    seed = seed_everything(global_args['seed'])

    dtype = dtype_convert(global_args['dtype'])
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset, updated_dataset_args = create_dataset(dataset_args, dtype, device)
    alphabet = dataset.alphabet
    inference_args['run_inference_args']['pred_gen_opts']['alphabet'] = alphabet
    size_dict = dataset.get_sizes()
    token_dict = dataset.get_ctrl_tokens()
    max_len_dict = dataset.get_max_seq_len()
    total_dict = {**size_dict, **token_dict, **max_len_dict}
    inference_args = specific_update(inference_args, total_dict)
    model_args = specific_update(model_args, total_dict)
    total_dict['seed'] = seed

    model, updated_model_args = create_model(model_args, dtype, device)
    model.to(dtype).to(device)

    #Find and load the best model checkpoint
    mod_ckpt_name = select_model(global_args['savedir'],
                                 inference_args['model_selection'])
    print(f"Using the model checkpoint {mod_ckpt_name}")
    best_model_ckpt = torch.load(mod_ckpt_name, map_location = device)
    model.load_state_dict(best_model_ckpt['model_state_dict'])

    if local_rank == 0:
        #Only do this under the first process
        tot_config = {
            'global_args' : global_args,
            'data' : updated_dataset_args,
            'model' : updated_model_args,
            'inference' : inference_args
        }
        save_completed_config('full_inference_config.yaml', tot_config, global_args['savedir'])
        save_token_size_dict(global_args['savedir'], total_dict, 'inference')

    #Set up dataloaders
    train_set, val_set, test_set = split_data_subsets(dataset, 
                                                    inference_args['splits'],
                                                    inference_args['train_size'],
                                                    inference_args['val_size'],
                                                    inference_args['test_size'])
    
    #Further divide the data here
    train_set = divide_parallel_subsets(train_set, n_procs, local_rank)
    val_set = divide_parallel_subsets(val_set, n_procs, local_rank)
    test_set = divide_parallel_subsets(test_set, n_procs, local_rank)
    
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_set, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              **inference_args['dloader_args'])
    val_loader = DataLoader(val_set, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              **inference_args['dloader_args'])
    test_loader = DataLoader(test_set, 
                              worker_init_fn=seed_worker,
                              generator=g,
                              **inference_args['dloader_args'])

    #Run inference
    print("Running inference...")
    if ('train' in inference_args['sets_to_run']):
        train_predictions = run_inference(model,
                                        train_loader,
                                        device=device, 
                                        **inference_args['run_inference_args']
                                        )
    else:
        train_predictions = None
    
    if ('val' in inference_args['sets_to_run']):
        val_predictions = run_inference(model,
                                        val_loader,
                                        device=device, 
                                        **inference_args['run_inference_args']
                                        )
    else:
        val_predictions = None
        
    if ('test' in inference_args['sets_to_run']):
        test_predictions = run_inference(model,
                                        test_loader,
                                        device=device, 
                                        **inference_args['run_inference_args']
                                        )
    else:
        test_predictions = None
        
    #save_inference_predictions(global_args['savedir'], 
    #                         train_predictions,
    #                         val_predictions,
    #                         test_predictions,
    #                         idx = local_rank)
    with open(f"{global_args['savedir']}/inference_predictions_{local_rank}.p", 'wb') as f:
        pickle.dump((train_predictions, val_predictions, test_predictions), f)

if __name__ == '__main__':
    main()
