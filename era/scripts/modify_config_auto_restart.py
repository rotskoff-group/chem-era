'''
Small module to modify the config file on restart. This is useful for when certain arguments
need to be added or changed on restart. 
'''
import argparse
import yaml
import os
import torch
from .top_level_utils import save_completed_config

def get_args() -> tuple[dict, str]:
    '''Parses the passed yaml file to get arguments'''
    parser = argparse.ArgumentParser(description = 'Modify NMR configuration file on training restart')
    parser.add_argument('config_file', type = str, help = 'The path to the YAML configuration file')
    args = parser.parse_args()
    return yaml.safe_load(open(args.config_file, 'r')), args.config_file

def main() -> None:
    '''
    Modifies the config file on restart.

    By design, this function can be executed repeatedly to modify the config file 
    but the file will only be modified under the following conditions:
        - The savedir exists
        - The savedir contains a checkpoint file called 'RESTART_checkpoint.pt' which contains 
            the relevant state dictionaries AND the previous epoch number
    
    The following modifications are made:
        - model->load_model is changed to RESTART_checkpoint.pt
        - training->prev_epochs is updated to the epoch in RESTART_checkpoint.pt +1
    
    The modified config file is used to overwrite the previous config file in the same directory
    '''
    current_config, config_filename = get_args()
    savedir = current_config['global_args']['savedir']
    if not os.path.isdir(savedir):
        print("Savedir does not exist. No modifications made, exiting.")
        return
    checkpoint_path = os.path.join(savedir, 'RESTART_checkpoint.pt')
    if not os.path.isfile(checkpoint_path):
        print("Restart checkpoint file does not exist. No modifications made, exiting.")
        return
    print("Directory and restart checkpoint detected, modifying config file...")
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')
    completed_epochs = checkpoint['epoch'] + 1
    current_config['training']['prev_epochs'] = completed_epochs
    current_config['model']['load_model'] = checkpoint_path 
    print("Saving modified configuration file and overwriting...")
    save_completed_config(config_filename,
                          current_config,
                          os.getcwd())
    print("Done!")
if __name__ == "__main__":
    main()