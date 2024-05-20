import torch
from era.data import create_dataset
import hydra
from era.scripts.top_level_utils import (
    seed_everything,
    dtype_convert
)
from functools import reduce
from era.data.tokenizer import BasicSmilesTokenizer
import era.data.input_generators as input_generators
import era.data.target_generators as target_generators
import h5py
import numpy as np
from typing import Callable
from multiprocessing import Pool
from tqdm import tqdm
from rdkit import Chem

def determine_alphabet(smiles: list[str], tokenizer: BasicSmilesTokenizer) -> list[str]:
    """Determines all unique tokens represented in a set of strings"""
    token_sets = [set(tokenizer.tokenize(smi)) for smi in tqdm(smiles)]
    final_set = list(reduce(lambda x, y: x.union(y), token_sets))
    alphabet = sorted(final_set)
    return alphabet

def run_process_parallel(f: Callable,
                         f_addn_args: dict,
                         num_processes: int,
                         *data_args):
    """Runs function f in parallel with num_processes processes

    Args:
        f: Function to be run in parallel
        f_addn_args: Additional keyword arguments required by f
        num_processes: Number of processes to run in parallel
        data_args: Arguments to be passed to f, should be a tuple of arrays. 
            
    Notes:     
        For data_args, if there is more than one array, then the arrays should have the 
        same first dimension size and index correspondence between elements.
        For multiple arrays, it is assumed they are passed in the order that the 
        elements would be passed to function f, i.e.

        ([x1, x2, x3, ...], [y1, y2, y3, ...], [z1, z2, z3, ...]) -> f(x1, y1, z1), f(x2, y2, z2), ...
    """
    assert(len(data_args) > 0)
    pool = Pool(processes = num_processes)
    if len(data_args) > 1:
        data_input = zip(*data_args)
        result = pool.starmap_async(f,
                                    data_input)
    else:
        result = pool.map_async(f, data_args[0])
        
    pool.close()
    pool.join()
    return result.get()



@hydra.main(version_base="1.3", config_path="../cfgs", config_name="precompute")
def main(cfg) -> None:
    dataset_args = cfg['precompute']
    global_args = cfg['global_args']
    dtype = dtype_convert(global_args['dtype'])
    device = None

    targets_h5 = h5py.File(dataset_args['target_file'], 'r')
    targets = targets_h5['targets']

    smiles = h5py.File(dataset_args['smiles_file'], 'r')['smiles']
    smiles = [Chem.CanonSmiles(smi.decode('utf-8')) for smi in smiles]
    tokenizer = BasicSmilesTokenizer()
    
    alp = dataset_args['alphabet']
    if alp is None:
        print("Determining alphabet based on SMILES")
        alphabet = determine_alphabet(smiles, tokenizer)
    else:
        print(f"Loading the following: {alp}")
        alphabet = np.load(alp, allow_pickle=True)
        alphabet = [str(x) for x in alphabet]

    input_generator = getattr(input_generators, dataset_args["input_generator"])(smiles, 
                                                                        tokenizer,
                                                                        alphabet, 
                                                                        **dataset_args["input_generator_addn_args"])
    
    target_generator = getattr(target_generators, dataset_args["target_generator"])(smiles, 
                                                                        tokenizer,
                                                                        alphabet,
                                                                        targets, 
                                                                        **dataset_args["target_generator_addn_args"])
    #Process the data in parallel
    num_processes = dataset_args['num_processes']
    print("Starting parallel runs...")
    # import pdb; pdb.set_trace()
    processed_inputs = run_process_parallel(input_generator.transform,
                                            {},
                                            num_processes,
                                            smiles)
    processed_targets = run_process_parallel(target_generator.transform,
                                            {},
                                            num_processes,
                                            smiles,
                                            targets)
    #Convert all targets and inputs into numpy arrays for encoding
    processed_inputs = np.array(list(map(lambda x : np.array(x), processed_inputs)))
    processed_targets = np.array(list(map(lambda x : np.array(x), processed_targets)))

    input_metadata = {
        'source_size' : input_generator.get_size(),
        'ctrl_tokens' : np.array(input_generator.get_ctrl_tokens()),
        'max_seq_len' : input_generator.get_max_seq_len()
    }
    target_metadata = {
        'target_size' : target_generator.get_size(),
        'ctrl_tokens' : np.array(target_generator.get_ctrl_tokens()),
        'max_seq_len' : target_generator.get_max_seq_len()
    }

    print(target_metadata)
    print(input_metadata)

    with h5py.File(dataset_args['output_file'], 'w') as f:
        f.create_dataset('inputs', data=processed_inputs)
        f.create_dataset('targets', data=processed_targets)
        f.create_dataset('smiles', data=smiles)
        f.create_dataset('alphabet', data=alphabet)
        inp_meta = f.create_group('input_metadata')
        for k, v in input_metadata.items():
            inp_meta.create_dataset(k, data=v)
        tar_meta = f.create_group('target_metadata')
        for k, v in target_metadata.items():
            tar_meta.create_dataset(k, data=v)
