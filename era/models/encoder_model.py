from era.networks import encoder, forward_fxns, embeddings
import torch
from torch import nn, Tensor
from typing import Callable, Optional

class EncoderModel(nn.Module):
    
    def __init__(self,
                 src_embed: str,
                 src_pad_token: int, 
                 src_forward_function: str,
                 pooler: str,
                 pooler_opts: dict,
                 output_head: str,
                 output_head_opts: dict,
                 d_model: int,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 d_out: int = 957,
                 source_size: int = 957,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 layer_norm_eps: float = 1e-05,
                 batch_first: bool = True,
                 enable_norm: bool = True,
                 norm_first: bool = False,
                 bias: bool = True,
                 num_layers: int = 6,
                 enable_nested_tensor: bool = True,
                 is_causal: bool = False,
                 permute_output: bool = False,
                 apply_masking: bool = False,
                 freeze_components: Optional[list] = None,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float
                 ):
        r"""See documentation for EncoderNetwork for description of parameters. The parameters replaced by
        string values are meant to be names of the compnents to be fetched using getattr"""
        super().__init__()
        if src_embed == 'mlp':
            src_embed_layer = embeddings.ProbabilityEmbedding(d_model)
        elif src_embed == 'matrix_scale':
            src_embed_layer = embeddings.MatrixScaleEmbedding(d_model, source_size)
        elif src_embed == 'spectra_continuous':
            src_embed_layer = embeddings.NMRContinuousEmbedding(d_model)
        elif src_embed == 'nn.embed':
            src_embed_layer = nn.Embedding(source_size, d_model, padding_idx = src_pad_token)
        elif src_embed is None:
            src_embed_layer = None
        else:
            raise ValueError(f"Unrecognized source embedding")
        
        pooler = getattr(encoder, pooler)
        output_head = getattr(encoder, output_head)

        src_fwd_fn = getattr(forward_fxns, src_forward_function)
        self.network = encoder.EncoderNetwork(
            src_embed=src_embed_layer,
            src_pad_token=src_pad_token,
            src_forward_function=src_fwd_fn,
            pooler=pooler,
            pooler_opts=pooler_opts,
            output_head=output_head,
            output_head_opts=output_head_opts,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            d_out=d_out,
            source_size=source_size,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            enable_norm=enable_norm,
            norm_first=norm_first,
            bias=bias,
            num_layers=num_layers,
            enable_nested_tensor=enable_nested_tensor,
            is_causal=is_causal,
            permute_output=permute_output,
            apply_masking=apply_masking,
            device=device,
            dtype=dtype
        )

        self.initialize_weights()
        self.freeze_components=freeze_components
        self.device=device
        self.dtype=dtype
    
    def initialize_weights(self) -> None:
        '''Initializes network weights'''
        for p in self.network.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def freeze(self) -> None:
        '''Disables greadients for specific network components'''
        if self.freeze_components is not None:
            for component in self.freeze_components:
                if hasattr(self.network, component):
                    for param in getattr(self.network, component).parameters():
                        param.requires_grad = False
    
    def forward(self, x: tuple[Tensor, tuple[str]]) -> Tensor:
        return self.network(x)
    
    def get_loss(self, 
                 x: tuple[Tensor, tuple[str]],
                 y: tuple[Tensor],
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return self.network.get_loss(x, y, loss_fn)
