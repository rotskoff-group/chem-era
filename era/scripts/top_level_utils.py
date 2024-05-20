import numpy as np
import random
import torch
import yaml
from typing import Union, Optional, Any
from torch.utils.data import Dataset
import os
import h5py
import pickle as pkl
from functools import reduce
import math
import warnings
from omegaconf import DictConfig, OmegaConf
import re

def seed_everything(seed: Union[int, None]) -> int:
    if seed is None:
        seed = torch.seed()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    #numpy requires the seed to be between 0 and 2**32 - 1
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def dtype_convert(dtype: str) -> torch.dtype:
    '''Maps string to torch dtype'''
    dtype_dict = {
        'float32': torch.float,
        'float64': torch.double,
        'float16': torch.half
    }
    return dtype_dict[dtype]

def save_completed_config(savename: str, config: Union[dict, DictConfig], savedir: str) -> None:
    '''Saves the completed config file to the savedir'''
    if not savename.endswith('.yaml'):
        savename = savename + '.yaml'
    with open(f"{savedir}/{savename}", 'w') as f:
        #The top level config dict may be a python dictionary, but 
        #   it may contain nested DictConfig objects. In this case, need to 
        #   convert the DictConfig objects to a dictionary before saving
        sanitized_config = {}
        for k, v in config.items():
            if isinstance(v, DictConfig):
                sanitized_config[k] = OmegaConf.to_container(v, resolve=True)
            else:
                sanitized_config[k] = v
        yaml.dump(sanitized_config, f, default_flow_style=False)

def split_data_subsets(dataset: Dataset,
                       splits: Optional[str],
                       train_size: float = 0.8,
                       val_size: float = 0.1,
                       test_size: float = 0.1) -> tuple[Dataset, Dataset, Dataset]:
    '''Splits the dataset using indices from passed file
    Args:
        dataset: The dataset to split
        splits: The path to the numpy file with the indices for the splits
        train_size: The fraction of the dataset to use for training
        val_size: The fraction of the dataset to use for validation
        test_size: The fraction of the dataset to use for testing
    '''
    if splits is not None:
        print(f"Splitting data using indices from {splits}")
        split_indices = np.load(splits, allow_pickle = True).item()
        train, val, test = split_indices['train'], split_indices['val'], split_indices['test']
        return torch.utils.data.Subset(dataset, train), torch.utils.data.Subset(dataset, val), \
            torch.utils.data.Subset(dataset, test)
    else:
        assert(train_size + val_size + test_size == 1)
        print(f"Splitting data using {train_size} train, {val_size} val, {test_size} test")
        train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
        return train, val, test
    
def divide_parallel_subsets(dataset: Dataset,
                            n_procs: int,
                            local_rank: int) -> Dataset:
    '''Takes a subset of the given dataset such that it is a subset that does not overlap with other n_proc - 1 subsets
    
    Note: Use math.ceil to ensure that the final process always gets the remainder of the dataset
    '''
    if n_procs == 1:
        start, end = 0, len(dataset)
        return torch.utils.data.Subset(dataset, range(start, end))
    else:
        assert(local_rank < n_procs)
        n_samples = len(dataset)
        subset_size = math.ceil(n_samples / n_procs)
        start, end = local_rank * subset_size, (local_rank + 1) * subset_size
        if end > n_samples:
            end = n_samples
        return torch.utils.data.Subset(dataset, range(start, end))
    
def save_train_history(savedir: str, loss_obj: tuple[list, ...]) -> None:
    """Saves the training history of the model to the specified directory
    
    Args:
        savedir: The directory to save the training history
        loss_obj: The tuple of lists containing the training history, with
            train_losses
            val_losses
            test_losses
            model_names
            best_losses
        in that order
    """
    train_losses, val_losses, test_losses, model_names, best_losses = loss_obj
    print("Saving losses")
    with h5py.File(f"{savedir}/losses.h5", "w") as f:
        f.create_dataset("train_losses", data = train_losses)
        f.create_dataset("val_losses", data = val_losses)
        f.create_dataset("test_losses", data = test_losses)
    
    with open(f"{savedir}/model_names_losses.pkl", "wb") as f:
        pkl.dump((model_names, best_losses), f)

def save_str_set(h5ptr: h5py.File, 
                 preds: list[tuple],
                 savename: str) -> None:
    """Saves string predictions to the passed h5 file pointer

    Args:
        h5ptr: The h5 file pointer to save to
        preds: The list of predictions, tuples of form (tgt, [pred, ...], smiles)
        savename: The name to save the predictions under

    In h5py, python strings are automatically encoded as variable-length UTF-8.
    It is assumed that each target has the same number of generated predictions
    """
    targets = [elem[0] for elem in preds]
    predictions = [elem[1] for elem in preds]
    group = h5ptr.create_group(savename)
    group.create_dataset("targets", data = targets)
    group.create_dataset("predictions", data = predictions)

def find_max_length(preds: list[list[np.ndarray]]) -> int:
    flattened_preds = list(reduce(lambda x, y : x + y, preds))
    max_len = max([len(elem) for elem in flattened_preds])
    return max_len

def pad_single_prediction(pred: list[np.ndarray],
                          max_len: int, 
                          pad_token: int) -> np.ndarray:
    fixed_seqs = []
    for elem in pred: 
        padded_elem = np.pad(elem, 
                             (0, max_len - len(elem)), 
                             'constant', 
                             constant_values = (pad_token,))
        fixed_seqs.append(padded_elem)
    return np.array(fixed_seqs)

def save_array_set(h5ptr: h5py.File, 
                   preds: list[tuple],
                   savename: str) -> None:
    """Saves array predictions to the passed h5 file pointer

    Args:
        h5ptr: The h5 file pointer to save to
        preds: The list of predictions, tuples of form ([tgt_i, ...], [pred_i, ...], [smiles_i, ...])
            where each tgt_i and pred_i is a 2D array and smiles_i is a 1D array. In the case where the targets
            are a 1D array, the targets are expanded to 2D. A similar procedure is applied to the predictions. 
        savename: The name to save the predictions under

    padding is done on these arrays to ensure size consistency when saving to hdf5
    """
    group = h5ptr.create_group(savename)
    targets = [elem[0] for elem in preds]
    targets = np.vstack(targets)
    predictions = [elem[1] for elem in preds]
    #Store smiles as a 1D list of strings which interface better with 
    #   h5py than Numpy's U-type string array which cannot be saved/converted
    #   to an h5 file
    smiles = []
    #Process SMILES
    for elem in preds:
        smi_elem = elem[2]
        if isinstance(smi_elem, list):
            smiles.extend(smi_elem)
        elif isinstance(smi_elem, str):
            smiles.append(smi_elem)
    #Process the predictions which may be lists of arrays
    if isinstance(predictions[0], list):
        max_len = find_max_length(predictions)
        pad_token = 999_999
        predictions = [
            np.expand_dims(
                pad_single_prediction(elem, max_len, pad_token), 
                axis = 0) for elem in predictions
        ]
        group.create_dataset("additional_pad_token", data = pad_token)
    predictions = np.vstack(predictions)
    assert(targets.shape[0] == predictions.shape[0] == len(smiles))
    group.create_dataset("targets", data = targets)
    group.create_dataset("predictions", data = predictions)
    group.create_dataset("smiles", data = smiles)

def save_inference_predictions(savedir: str,
                               train_predictions: list,
                               val_predictions: list,
                               test_predictions: list, 
                               idx: int = None) -> None:
    '''Saves the predictions from inference as h5 files in the specified directory

    Args:
        savedir: The directory to save to
        train_predictions: The list of train predictions
        val_predictions: The list of validation predictions
        test_predictions: The list of test predictions
        idx: The index to distinguish the predictions from different parallel runs

    The formatting for the h5py file changes depending on the form of the predictions. 
    However, at the top level, the following are exposed: 'train', 'val', and 'test'. Depending
    on if predictions are passed for each set, each key further has the two subkeys 'target' and
    'predictions'.
    .
    '''    
    print("Saving predictions...")
    test_element = test_predictions[0]
    assert(isinstance(test_element, tuple))
    filehandle = f"{savedir}/predictions.h5" if idx is None else f"{savedir}/predictions_{idx}.h5"
    with h5py.File(filehandle, 'w') as f:    
        if isinstance(test_element[0], str):
            save_fxn = save_str_set
        elif isinstance(test_element[0], np.ndarray):   
            save_fxn = save_array_set
        if train_predictions is not None:
            save_fxn(f, train_predictions, 'train')
        if val_predictions is not None:
            save_fxn(f, val_predictions, 'val')
        if test_predictions is not None:
            save_fxn(f, test_predictions, 'test')

def save_token_size_dict(savedir: str,
                         token_size_dict: dict,
                         savetag: str) -> None:
    with open(f"{savedir}/{savetag}_token_size_dict.pkl", "wb") as f:
        pkl.dump(token_size_dict, f)


def specific_update(mapping: DictConfig[str, Any], update_map: dict[str, Any]) -> dict[str, Any]:
    """Recursively update keys in a mapping with the values specified in update_map"""
    mapping = mapping.copy()
    for k, v in mapping.items():
        #Give precedence to existing parameter settings
        if (k in update_map) and (not isinstance(v, DictConfig)) and (v is None):
            mapping[k] = update_map[k]
        elif isinstance(v, DictConfig):
            mapping[k] = specific_update(v, update_map)
    return mapping

def extract_loss_val(checkpoint_name: str) -> float:
    """Extract loss value from a checkpoint name"""
    return float(re.findall(r"\d+\.\d+", checkpoint_name)[0])

def select_model(savedir: str,
                 criterion: str) -> str:
    '''Selects the saved model checkpoints based on the criterion
    
    Args:
        savedir: Directory where model checkpoints are saved
        critierion: Either 'lowest' or 'highest' loss, or name of a model checkpoint to use
    '''
    if criterion not in ['lowest', 'highest']:
        warnings.warn("Assuming passed value is a model checkpoint")
        return criterion
    if not os.path.isfile(f"{savedir}/model_names_losses.pkl"):
        print("Deducing checkpoint losses from file names")
        all_files = os.listdir(savedir)
        checkpoint_files = list(filter(lambda x : x.endswith('.pt') and x != 'RESTART_checkpoint.pt', all_files))
        model_names = [f"{savedir}/{f}" for f in checkpoint_files]
        best_losses = [extract_loss_val(f) for f in checkpoint_files]
    else:
        print("Loading checkpoint losses from file")
        with open(f"{savedir}/model_names_losses.pkl", "rb") as f:
            model_names, best_losses = pkl.load(f)
    if criterion == 'lowest':
        idx = np.argmin(best_losses)
    elif criterion == 'highest':
        idx = np.argmax(best_losses)
    return model_names[idx]
