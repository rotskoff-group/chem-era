import numpy as np
from .tokenizer import BasicSmilesTokenizer

def look_ahead_smiles(smiles: list[str], tokenizer: BasicSmilesTokenizer) -> int:
    """Determines the maximum length of the smiles strings in numbers of tokens"""
    max_len = 0
    for i in range(len(smiles)):
        tokens = tokenizer.tokenize(smiles[i])
        max_len = max(max_len, len(tokens))
    #To account for additional stop token 
    return max_len + 1 

#Base class for input generators, to be inherited by others
class TargetGeneratorBase:
    def __init__(self,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 targets: np.ndarray):
        self.alphabet_size = -100
        self.max_len = -100
        self.pad_token = -100
        self.stop_token = -100
        self.start_token = -100
    
    def transform(self, smiles: str, targets: float) -> float:
        pass

    def get_size(self) -> int:
        return self.alphabet_size
    
    def get_ctrl_tokens(self) -> list[int, int, int]:
        return [self.stop_token, self.start_token, self.pad_token]
    
    def get_max_seq_len(self) -> int:
        return self.max_len

class ScalarTarget(TargetGeneratorBase):
    """Process scalar labels into a tensor"""
    def __init__(self, 
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 targets: np.ndarray):
        """
        Args:
            targets: Array of scalar labels
        """
        super().__init__(smiles, tokenizer, alphabet, targets)


    def transform(self, smiles: str, targets: float) -> tuple[float]:
        return (targets,)

class SMILESTarget(TargetGeneratorBase):
    """Process SMILES strings into a tokenized array with padding to maximum sequence length"""
    def __init__(self, 
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray,
                 targets: np.ndarray):
        """
        Args:
            smiles: Array of SMILES strings
            tokenizer: Instance of BasicSmilesTokenizer
            alphabet: Array of SORTED unique tokens
            targets: Array of scalar targets

        Converts a SMILES string into a right-padded sequence of tokens. The padding token 
        is taken as the length of the alphabet. For consistency with the SMILES input generator, the 
        control tokens are:
            pad: len(alphabet)
            start: len(alphabet) + 1
            stop: len(alphabet) + 2
        """
        super().__init__(smiles, tokenizer, alphabet, targets)
        self.pad_token = len(alphabet)
        self.start_token = len(alphabet) + 1
        self.stop_token = len(alphabet) + 2
        self.tokenizer = tokenizer
        self.max_len = look_ahead_smiles(smiles, self.tokenizer)
        self.index_map = {char: i for i, char in enumerate(alphabet)}
        #Accounting for padding, start, and stop tokens in the alphabet.
        self.alphabet_size = len(alphabet) + 3
    
    def transform(self, smiles: str, targets: float) -> tuple[np.ndarray]:
        smiles = str(smiles)
        tokenized_smiles = self.tokenizer.tokenize(smiles)
        tokenized_smiles = [self.index_map[char] for char in tokenized_smiles]
        tokenized_smiles = tokenized_smiles + [self.stop_token]
        #Pad to the maximum length
        tokenized_smiles = tokenized_smiles + [self.pad_token] * (self.max_len - len(tokenized_smiles))
        return (np.array(tokenized_smiles),)
