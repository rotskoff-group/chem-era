import torch
from torch import nn, Tensor
import math
from typing import Optional, Any, Tuple, Callable

class PositionalEncoding(nn.Module):

    """ Positional encoding with option of selecting specific indices to add """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 30000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor, ind: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            ind: Tensor, shape [batch_size, seq_len] or NoneType
        '''
        #Select and expand the PE to be the right shape first
        if ind is not None:
            added_pe = self.pe[torch.arange(1).reshape(-1, 1), ind, :]
            x = x + added_pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class Transformer(nn.Module):

    model_id = "Transformer"
    
    def __init__(self, src_embed: nn.Module, tgt_embed: nn.Module,
                 src_pad_token: int, tgt_pad_token: int, 
                 src_forward_function: Callable[[Tensor, nn.Module, int, Optional[nn.Module]], Tuple[Tensor, Optional[Tensor]]],
                 tgt_forward_function: Callable[[Tensor, nn.Module, int, Optional[nn.Module]], Tuple[Tensor, Optional[Tensor]]],
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu', custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None, target_size: int = 50, source_size: int = 957,
                 layer_norm_eps: float = 1e-05, batch_first: bool = True, norm_first: bool = False, 
                 device: torch.device = None, dtype: torch.dtype = torch.float):
        
        r"""Most parameters are standard for the PyTorch transformer class. A few specific ones that have been added:

        src_embed: The embedding module for the src tensor passed to the model
        tgt_embed: The embedding module for the tgt tensor passed to the model
        src_pad_token: The index used to indicate padding in the source sequence
        tgt_pad_token: The index used to indicate padding in the target sequence
        src_forward_function: A function that processes the src tensor using the src embedding, src pad token, and positional encoding to generate
            the embedded src and the src_key_pad_mask
        tgt_forward_function: A function that processes the tgt tensor using the tgt embedding, tgt pad token, and positional encoding to generate
            the embedded tgt and the tgt_key_pad_mask
        target_size (int): Size of the target alphabet (including start, stop, and pad tokens).
        source_size (int): Size of the source alphabet (including start, stop, and pad tokens).
        batch_first is set to be default True, more intuitive to reason about dimensionality if batching 
            dimension is first
        """
        super().__init__()

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_fwd_fn = src_forward_function
        self.tgt_fwd_fn = tgt_forward_function
        self.src_pad_token = src_pad_token
        self.tgt_pad_token = tgt_pad_token

        self.tgt_size = target_size
        self.src_size = source_size
        self.d_model = d_model
        self.dtype = dtype
        self.device = device

        self.transformer = nn.Transformer(d_model = d_model, nhead = nhead, num_encoder_layers = num_encoder_layers,
                                          num_decoder_layers = num_decoder_layers, dim_feedforward = dim_feedforward,
                                          dropout = dropout, activation = activation, custom_encoder = custom_encoder,
                                          custom_decoder = custom_decoder, layer_norm_eps = layer_norm_eps, 
                                          batch_first = batch_first, norm_first = norm_first, device = device, 
                                          dtype = self.dtype)
        
        #Always use start and stop tokens for target, only padding is optional
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.out = nn.Linear(d_model, target_size)

    def _get_tgt_mask(self, size):
        #Generate a mask for the target to preserve autoregressive property. Note that the mask is 
        #   Additive for the PyTorch transformer
        mask = torch.tril(torch.ones(size, size, dtype = self.dtype, device = self.device))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask
    
    def _sanitize_forward_args(self, 
                               x: Tuple[Tensor, Tuple],
                               y: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        '''Processes raw x, y inputs to the transformer for use in forward()
        Args:
            x: Tuple of a tensor (input) and the set of smiles strings (smiles)
            y: A tuple of a the shifted target tensor and full target tensor
        '''
        #Unpack the tuples
        inp, _ = x
        shifted_y, _ = y
        if isinstance(self.src_embed, nn.Embedding):
            inp = inp.long()
        if isinstance(self.tgt_embed, nn.Embedding):
            shifted_y = shifted_y.long()
        return inp, shifted_y
    
    #Sketch of what the forward function could look like with more abstraction
    def forward(self, 
                x: Tuple[Tensor, Tuple], 
                y: Tuple[Tensor, Tensor]) -> Tensor:
        src, tgt = self._sanitize_forward_args(x, y)
        tgt_mask = self._get_tgt_mask(tgt.size(1)).to(src.device)
        src_embedded, src_key_pad_mask = self.src_fwd_fn(src, self.d_model, self.src_embed, self.src_pad_token, self.pos_encoder)
        tgt_embedded, tgt_key_pad_mask = self.tgt_fwd_fn(tgt, self.d_model, self.tgt_embed, self.tgt_pad_token, self.pos_encoder)
        transformer_out = self.transformer(src_embedded, tgt_embedded,
                                           tgt_mask = tgt_mask,
                                           tgt_key_padding_mask = tgt_key_pad_mask,
                                           #NOTE: src_key_padding_mask causes issues when dealing with no_grad() context, see https://github.com/pytorch/pytorch/issues/106630
                                           src_key_padding_mask = src_key_pad_mask)
        out = self.out(transformer_out)
        return out

    def get_loss(self,
                 x: Tuple[Tensor, Tuple], 
                 y: Tuple[Tensor], 
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        """
        Unpacks the input and obtains the loss value
        Args:
            x: Tuple of a tensor (input) and the set of smiles strings (smiles)
            y: A tuple of a the shifted target tensor and full target tensor
            loss_fn: The loss function to use for the model, with the signature
                tensor, tensor -> tensor
        """
        pred = self.forward(x, y)
        pred = pred.permute(0, 2, 1)
        _, full_y = y
        if isinstance(self.tgt_embed, nn.Embedding):
            full_y = full_y.long()
        loss = loss_fn(pred, full_y.to(self.device))
        return loss