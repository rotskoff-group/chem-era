"""Generic class for models that use RDKit for evaluation"""

from era.networks import encoder, forward_fxns, embeddings
import era.models.rdkit_model as rdkit_model
import torch
from torch import nn, Tensor
from typing import Callable, Optional
import numpy as np
from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem import Lipinski
from rdkit import Chem as chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import math
from rdkit import DataStructs
RDLogger.DisableLog('rdApp.*')

def validity_logit(smi: str) -> float:
    raise NotImplementedError
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        return 100
    else:
        return -100

def QED_energy(smi: str) -> float:
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        p = QED.qed(mol)
        return -np.log(p)
    else: 
        #Based on torch.log of 0.01
        return 4.5

class RDKitModel(nn.Module):

    def __init__(self,
                 alphabet: np.ndarray,
                 criterion: str,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None,
                 *args, **kwargs):
        super().__init__()
        self.alphabet = alphabet
        if isinstance(self.alphabet, str):
            self.alphabet = np.load(self.alphabet, allow_pickle=True)
        self.pad_token = len(self.alphabet)
        self.start_token = len(self.alphabet) + 1
        self.stop_token = len(self.alphabet) + 2
        #print(dir())
        #print(criterion)
        #assert criterion in dir()
        self.criterion = eval(criterion)
        self.dtype = dtype
        self.device = device

    def forward(self, x: tuple[Tensor]) -> Tensor:
        '''x: (n_smi, max_len)'''
        x = x[0]
        values = [] #Will become n_smi
        for i in range(x.shape[0]):
            smi_tokens = x[i, :]
            smi_tokens = self._strip_control_tokens(smi_tokens)
            smi_tokens = smi_tokens[smi_tokens != self.pad_token]
            smi = ''.join([self.alphabet[j] for j in smi_tokens])
            val = self.criterion(smi)
            values.append(val)
        return torch.tensor(values, dtype = self.dtype, device = self.device).unsqueeze(-1)

    def _strip_control_tokens(self, prompt: Tensor):
        prompt = prompt[prompt != self.pad_token]
        prompt = prompt[prompt != self.start_token]
        prompt = prompt[prompt != self.stop_token]
        return prompt.cpu().numpy().astype(int)
    
    def freeze(self):
        return None

def tanimoto_energy(prompt_smi: str, tst_smi: str) -> float:
    prompt_mol = chem.MolFromSmiles(prompt_smi)
    assert prompt_mol is not None
    tst_mol = chem.MolFromSmiles(tst_smi)
    if tst_mol is not None:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(prompt_mol, 2, nBits=1024)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(tst_mol, 2, nBits=1024)
        p = DataStructs.TanimotoSimilarity(fp1, fp2)
        if p == 0:
            return 10
        elif p == 1:
            return 3.5
        else:
            return -np.log(p)
    else:
        return 10

class SimilarityModel(nn.Module):

    def __init__(self,
                 alphabet: np.ndarray,
                 criterion: str,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        super().__init__()
        self.alphabet = alphabet
        if isinstance(self.alphabet, str):
            self.alphabet = np.load(self.alphabet, allow_pickle=True)
        self.pad_token = len(self.alphabet)
        self.start_token = len(self.alphabet) + 1
        self.stop_token = len(self.alphabet) + 2
        self.criterion = eval(criterion)
        self.dtype = dtype
        self.device = device
    
    def forward(self, x: tuple[Tensor]):
        '''x: a tuple containing:
            prompts: (n_smi, max_len)
            samples: (n_smi, max_len)
        '''
        prompts, samples = x[0]
        values = []
        for i in range(prompts.shape[0]):
            curr_prompt_tokens = prompts[i, :]
            curr_prompt_str_tokens = self._strip_control_tokens(curr_prompt_tokens)
            curr_prompt_smi = ''.join([self.alphabet[j] for j in curr_prompt_str_tokens])
            curr_sample = samples[i, :]
            curr_sample_str_tokens = self._strip_control_tokens(curr_sample)
            curr_sample_smi = ''.join([self.alphabet[j] for j in curr_sample_str_tokens])
            val = self.criterion(curr_prompt_smi, curr_sample_smi)
            values.append(val)
        return torch.tensor(values, dtype = self.dtype, device = self.device).unsqueeze(-1)

    def _strip_control_tokens(self, prompt: Tensor):
        prompt = prompt[prompt != self.pad_token]
        prompt = prompt[prompt != self.start_token]
        prompt = prompt[prompt != self.stop_token]
        return prompt.cpu().numpy().astype(int)
    
    def freeze(self):
        pass


def calc_LogP(smi: str) -> float:
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        return Crippen.MolLogP(mol)
    else:
        return None
    
def calc_MR(smi: str) -> float:
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        return Crippen.MolMR(mol)
    else:
        return None
    
def calc_CSP3(smi: str) -> float:
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        return Lipinski.FractionCSP3(mol)
    else:
        return None

def calc_NHeavy(smi: str) -> int:
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        return Lipinski.HeavyAtomCount(mol)
    else:
        return None
    
def calc_NumRings(smi: str) -> int:
    mol = chem.MolFromSmiles(smi)
    if mol is not None:
        return Lipinski.RingCount(mol)
    else:
        return None

class GaussianModel(nn.Module):
    """
    Defines a Gaussian centered around a target value and a user-defined standard deviation
    The failure_value is returned in the case of a SMILES that could not be parsed successfully
    """
    def __init__(self, 
                 alphabet: np.ndarray,
                 criterion: str,
                 mu: float,
                 sigma: float,
                 failure_value: float,
                 prompted: bool = False,
                 dtype: torch.dtype = torch.float,
                 device: torch.device = None):
        super().__init__()
        self.alphabet = alphabet
        if isinstance(self.alphabet, str):
            self.alphabet = np.load(self.alphabet, allow_pickle=True)
        self.pad_token = len(self.alphabet) 
        self.start_token = len(self.alphabet) + 1
        self.stop_token = len(self.alphabet) + 2
        self.prompted = prompted
        self.criterion = eval(criterion)
        self.mu = mu
        self.fail_val = failure_value
        self.sigma = sigma
        self.dtype = dtype
        self.device = device
    
    def get_logp(self, x):
        return -0.5 * ((x - self.mu) / self.sigma)**2
    
    def _strip_control_tokens(self, prompt: Tensor):
        prompt = prompt[prompt != self.pad_token]
        prompt = prompt[prompt != self.start_token]
        prompt = prompt[prompt != self.stop_token]
        return prompt.cpu().numpy().astype(int)
    
    def forward(self, x: tuple[Tensor]):
        """
        Behavior changes depending on if prompted or not. If prompted, 
        then the model computes the value over the generation only
        """
        if self.prompted:
            _, samples = x[0]
        else:
            samples = x[0]
        values = []
        for i in range(samples.shape[0]):
            curr_sample = samples[i, :]
            curr_sample_str_tokens = self._strip_control_tokens(curr_sample)
            curr_sample_smi = ''.join([self.alphabet[j] for j in curr_sample_str_tokens])
            val = self.criterion(curr_sample_smi)
            if val is not None:
                energy = -self.get_logp(val)
            else:
                energy = self.fail_val
            values.append(energy)
        return torch.tensor(values, dtype = self.dtype, device = self.device).unsqueeze(-1)
    
    def freeze(self):
        pass
        
