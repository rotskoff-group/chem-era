from torch.utils.data import Dataset
import numpy as np
import h5py
from era.data.tokenizer import BasicSmilesTokenizer
from functools import reduce
from rdkit import Chem as chem
import era.data.input_generators as input_generators
import era.data.target_generators as target_generators
import torch
from tqdm import tqdm

#TODO: migrate this logic to precompute script, axe this class
class SMILESDatasetPrecompute:
    """Dataset that processes SMILES as x and targets as y"""

    def __init__(self,
                 smiles_file: str,
                 target_file: str,
                 input_generator: str,
                 input_generator_addn_args: dict,
                 target_generator: str,
                 target_generator_addn_args: dict,
                 input_outname: str,
                 target_outname: str,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        """
        Args:
            smiles_file: Path to the HDF5 file with smiles
            target_file: Path to the HDF5 file with target values
            input_generator: Function name that generates the model input
            input_generator_addn_args: Additional arguments for the input generator
            target_generator: Function name that generates the model target
            target_generator_addn_args: Additional arguments for the target generator
            input_outname: Name of the output file for the input data
            target_outname: Name of the output file for the target data
        """
        self.targets_h5 = h5py.File(target_file, 'r')
        self.targets = self.targets_h5['targets']

        self.smiles = h5py.File(smiles_file, 'r')['smiles']
        self.tokenizer = BasicSmilesTokenizer()

        self.device = device
        self.dtype = dtype

        #Alphabet determined first
        self.smiles = [smi.decode('utf-8') for smi in tqdm(self.smiles)]
        self.smiles = [chem.CanonSmiles(smi) for smi in tqdm(self.smiles)]
        self.alphabet = self._determine_smiles_alphabet(self.smiles)
        np.save("alphabet.npy", self.alphabet)

        
        # Initialize the input and target generators with a quick "look ahead" to
        #   determine global information, i.e. max lengths and padding
        self.input_generator = getattr(input_generators, input_generator)(self.smiles, self.tokenizer,
                                                                          self.alphabet, **input_generator_addn_args)
        self.target_generator = getattr(target_generators, target_generator)(self.smiles, self.tokenizer, 
                                                                             self.alphabet, self.targets,
                                                                             **target_generator_addn_args)

        # print("Preprocessing data")
        # self.preprocessed_model_inputs = []
        # self.preprocessed_model_targets = []
        # for i in tqdm(range(len(self))):
        #     smiles_data = self.smiles[i]
        #     target_data = self.targets[i]
        #     model_input = self.input_generator.transform(smiles_data)
        #     model_target = self.target_generator.transform(smiles_data, 
        #                                                    target_data)
            
        #     model_input = torch.from_numpy(model_input).to(self.dtype).to(self.device)
        #     model_target = tuple([torch.tensor(elem).to(self.dtype).to(self.device) 
        #                           for elem in model_target])
        #     self.preprocessed_model_inputs.append(model_input)
        #     self.preprocessed_model_targets.append(model_target)

    def _determine_smiles_alphabet(self, smiles: list[str]):
        """Generates the alphabet from the set of smiles strings
        Args:
            smiles: list of smiles strings
        """
        token_sets = [set(self.tokenizer.tokenize(smi)) for smi in smiles]
        final_set = list(reduce(lambda x, y: x.union(y), token_sets))
        alphabet = sorted(final_set)
        return alphabet

    def get_sizes(self) -> dict[int, int]:
        """Returns the padding tokens for the input and target"""
        input_size = self.input_generator.get_size()
        target_size = self.target_generator.get_size()
        return {'source_size': input_size, 'target_size': target_size}

    def get_ctrl_tokens(self) -> dict[str, int]:
        """Returns the stop, start, and pad tokens for the input and target as dicts"""
        input_tokens = self.input_generator.get_ctrl_tokens()
        target_tokens = self.target_generator.get_ctrl_tokens()
        src_stop_token, src_start_token, src_pad_token = input_tokens
        tgt_stop_token, tgt_start_token, tgt_pad_token = target_tokens
        token_dict = {
            'src_stop_token': src_stop_token,
            'src_start_token': src_start_token,
            'src_pad_token': src_pad_token,
            'tgt_stop_token': tgt_stop_token,
            'tgt_start_token': tgt_start_token,
            'tgt_pad_token': tgt_pad_token
        }
        return token_dict

    def get_max_seq_len(self) -> dict[str, int]:
        """Returns the max sequence length for the input and target as dicts"""
        input_len = self.input_generator.get_max_seq_len()
        target_len = self.target_generator.get_max_seq_len()
        return {'max_src_len': input_len, 'max_tgt_len': target_len}
    
    def process_data(self):
        """Preprocesses the data and stores it in memory"""
        self.preprocessed_model_inputs = []
        self.preprocessed_model_targets = []
        for i in tqdm(range(len(self))):
            smiles_data = self.smiles[i]
            target_data = self.targets[i]
            model_input = self.input_generator.transform(smiles_data)
            model_target = self.target_generator.transform(smiles_data, 
                                                           target_data)
            self.preprocessed_model_inputs.append(model_input)
            self.preprocessed_model_targets.append(model_target)
        
class SMILESDataset(Dataset):

    def __init__(self, 
                 data_file: str,
                 load_to_memory: bool = False,
                 inference_mode: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        """
        Dataset for training models assuming precomputed data
        Args:
            data_file: Path to a precomputed dataset stored as an hdf5 dataset
            load_to_memory: If true, the data is loaded into CPU memory at the start of training
            dtype: Data type for tensors
            device: Device for tensors
        
        Note that all relevant metadata (padding token indices, max sequence lengths, etc.) are expected to be 
        stored under the 'metadata' group in the respective HDF5 files.
        """
        self.load_to_memory = load_to_memory
        self.inference_mode = inference_mode
        self.device = device
        self.dtype = dtype
        data_ptr = h5py.File(data_file, 'r')
        self.alphabet = data_ptr['alphabet'][()]
        self.alphabet = [x.decode('utf-8') for x in self.alphabet]
        #Retrieve metadata information
        self.length = len(data_ptr['smiles'])
        self.input_metadata = self.extract_metadata(data_ptr['input_metadata'])
        self.target_metadata = self.extract_metadata(data_ptr['target_metadata'])
        self.data_path = data_file
        #Delete open file objects
        del data_ptr

    def __len__(self):
        return self.length  
    
    def open_hdf5(self):
        self.data_hdf5 = h5py.File(self.data_path, 'r')
        self.inputs = self.data_hdf5['inputs']
        self.targets = self.data_hdf5['targets']
        self.smiles = self.data_hdf5['smiles']
        self.data_opened = True

        if self.load_to_memory:
            self.inputs = self.inputs[()]
            self.targets = self.targets[()]
            self.smiles = self.smiles[()]
    
    def __getitem__(self, idx):
        if not hasattr(self, 'data_opened'):
            self.open_hdf5()
        
        if not self.inference_mode:
            model_input = torch.from_numpy(self.inputs[idx]).to(self.dtype).to(self.device)
        else:
            model_input = torch.from_numpy(np.array([self.input_metadata['ctrl_tokens'][1]])).to(self.dtype).to(self.device)

        model_target = tuple([torch.tensor(elem).to(self.dtype).to(self.device) 
                                   for elem in self.targets[idx]])

        return (model_input, self.smiles[idx]), model_target
    
    def extract_metadata(self, metadata_hdf5: h5py.Group) -> dict:
        '''Extracts a hdf5 group as a dictionary for storage in memory
        Also converts numpy datatypes into the corresponding python datatypes
        '''
        meta_dict = {}
        for key in metadata_hdf5.keys():
            value = metadata_hdf5[key][()]
            if value.ndim == 0:
                value = int(value)
            else:
                value = value.tolist()
            meta_dict[key] = value
        return meta_dict
    
    def get_sizes(self) -> dict[int, int]:
        """Returns the padding tokens for the input and target"""
        input_size = self.input_metadata['source_size']
        target_size = self.target_metadata['target_size']
        return {'source_size': input_size, 'target_size': target_size}

    def get_ctrl_tokens(self) -> dict[str, int]:
        """Returns the stop, start, and pad tokens for the input and target as dicts"""
        input_tokens = self.input_metadata['ctrl_tokens']
        target_tokens = self.target_metadata['ctrl_tokens']
        src_stop_token, src_start_token, src_pad_token = input_tokens
        tgt_stop_token, tgt_start_token, tgt_pad_token = target_tokens
        token_dict = {
            'src_stop_token': src_stop_token,
            'src_start_token': src_start_token,
            'src_pad_token': src_pad_token,
            'tgt_stop_token': tgt_stop_token,
            'tgt_start_token': tgt_start_token,
            'tgt_pad_token': tgt_pad_token
        }
        return token_dict

    def get_max_seq_len(self) -> dict[str, int]:
        """Returns the max sequence length for the input and target as dicts"""
        input_len = self.input_metadata['max_seq_len']
        target_len = self.target_metadata['max_seq_len']
        return {'max_src_len': input_len, 'max_tgt_len': target_len}
    
    def get_weights(self, indices) -> np.array:
        '''Returns the weights for the dataset'''
        sorted_inds = np.sort(indices)
        data_ptr = h5py.File(self.data_path, 'r')
        #Compute the weight based only on the training population 
        #   frequencies to avoid leakage
        sub_targs = data_ptr['targets'][sorted_inds]
        total_0 = np.sum(sub_targs[()] == 0)
        total_1 = np.sum(sub_targs[()] == 1)
        total = total_0 + total_1
        weight_0 = total_0 / total
        weight_1 = total_1 / total
        weights = np.array(
            [weight_0 if elem == 0 else weight_1 for elem in data_ptr['targets'][sorted_inds]]
        )
        del data_ptr
        return weights
