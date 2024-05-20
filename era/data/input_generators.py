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
class InputGeneratorBase:
    #Getters have concrete implementations, but constructor and transform are not implemented
    def __init__(self,
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray):
        pass

    def transform(self, smiles: str) -> np.ndarray:
        pass

    def get_size(self) -> int:
        return self.alphabet_size
    
    def get_ctrl_tokens(self) -> list[int, int, int]:
        return [self.stop_token, self.start_token, self.pad_token]
    
    def get_max_seq_len(self) -> int:
        return self.max_len

class SMILESInputBasic(InputGeneratorBase):
    """Process SMILES strings into a tokenized array with padding"""
    def __init__(self, 
                 smiles: np.ndarray,
                 tokenizer: BasicSmilesTokenizer,
                 alphabet: np.ndarray, 
                 generative_mode: bool = False):
        """
        Args:
            smiles: Array of SMILES strings
            tokenizer: Instance of BasicSmilesTokenizer
            alphabet: Array of SORTED unique tokens
            generative_mode: If the model is being trained to generate SMILES strings. 
                This does two things:
                    1) Shifts sequences by adding a start token
                    2) Sets the stop token (only used in the target generator)
                The stop token is set here because the size of the source alphabet needs 
                to be known at this stage.

        Converts a SMILES string into a right-padded sequence of tokens. The padding token 
        is taken as the length of the alphabet. 

        Shifting example:
        Given a sequence of tokens with padding token 0:
            [A, B, C, 0, 0, 0]
        Shifting adds a start token and shifts the sequence to the right:
            [<start>, A, B, C, 0, 0]
        The corresponding target for this sequence will be:
            [A, B, C, <EOS>, 0, 0]
        Note that the lengths of both sequences are the same. The EOS token is only used in the target
            generator. For consistency between the two, the tokens are:
                pad: len(alphabet)
                start: len(alphabet) + 1
                stop: len(alphabet) + 2
        """
        self.pad_token = len(alphabet)
        self.tokenizer = tokenizer
        self.max_len = look_ahead_smiles(smiles, self.tokenizer)
        self.index_map = {char: i for i, char in enumerate(alphabet)}
        self.alphabet_size = len(alphabet) + 1 #Accounting for padding
        self.generative = generative_mode
        if self.generative:
            self.start_token = len(alphabet) + 1
            self.stop_token = len(alphabet) + 2
            self.alphabet_size += 2
        else:
            self.start_token = -100
            self.stop_token = -100

    def transform(self, smiles: str) -> np.ndarray:
        smiles = str(smiles) #Type cast for safety
        tokenized_smiles = self.tokenizer.tokenize(smiles)
        tokenized_smiles = [self.index_map[char] for char in tokenized_smiles]
        if self.generative:
            tokenized_smiles = [self.start_token] + tokenized_smiles
        #Pad to the maximum length
        tokenized_smiles = tokenized_smiles + [self.pad_token] * (self.max_len - len(tokenized_smiles))
        return np.array(tokenized_smiles)