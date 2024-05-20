import torch
from torch import Tensor
from torch import nn

class ProbabilityEmbedding(nn.Module):
    """ MLP based connection between substructure predictions and transformer. """

    def __init__(self, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (batch_size, seq_len, 1)
        '''
        return self.layers(x)

class MatrixScaleEmbedding(nn.Module):
    """ Broadcast matrix multiplication connection between substructure predictions and transformer. """

    def __init__(self, d_model: int, n_substructures: int = 957):
        super().__init__()
        self.n_substructures = n_substructures 
        self.layers = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.n_substructures, d_model)))
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        x: (batch_size, seq_len, 1)
        '''
        return x * self.layers
    
class NMRContinuousEmbedding(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 1):
        '''
        For implementation simplicity, only use one head for now,
            extending to multiple heads is trivial.
        '''
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Linear(2, d_model // num_heads) for _ in range(num_heads)
        ])
    
    def forward(self, x):
        '''
        x: Tensor, shape (batch_size, seq_len, 2)
        Returns the embedded tensor (batch_size, seq_len, d_model)
        '''
        out = [head(x) for head in self.heads]
        return torch.cat(out, dim = -1)