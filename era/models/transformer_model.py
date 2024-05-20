from era.networks import transformer, forward_fxns, embeddings
import torch
from torch import nn, Tensor
from typing import Tuple, Callable, Optional, Any


class TransformerModel(nn.Module):
    """ Example model wrapper for transformer network """

    def __init__(self, src_embed: str,
                 tgt_embed: str,
                 src_pad_token: int,
                 tgt_pad_token: int,
                 src_forward_function: str,
                 tgt_forward_function: str,
                 freeze_components: Optional[list] = None,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 custom_encoder: Optional[Any] = None,
                 custom_decoder: Optional[Any] = None,
                 target_size: int = 50,
                 source_size: int = 957,
                 layer_norm_eps: float = 1e-05,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 device: torch.device = None,
                 dtype: torch.dtype = torch.float):
        r"""Most parameters are standard for the PyTorch transformer class. A few specific ones that have been added:

        src_embed: The name of the embedding module for the src tensor passed to the model
        tgt_embed: The name of the embedding module for the tgt tensor passed to the model
        src_pad_token: The index used to indicate padding in the source sequence
        tgt_pad_token: The index used to indicate padding in the target sequence
        src_forward_function: Name of the function that processes the src tensor using the src embedding, src pad token, and positional encoding to generate
            the embedded src and the src_key_pad_mask
        tgt_forward_function: Name of the function that processes the tgt tensor using the tgt embedding, tgt pad token, and positional encoding to generate
            the embedded tgt and the tgt_key_pad_mask
        freeze_components: List of component names to freeze
        target_size (int): Size of the target alphabet (including start, stop, and pad tokens).
        source_size (int): Size of the source alphabet (including start, stop, and pad tokens).
        batch_first is set to be default True, more intuitive to reason about dimensionality if batching 
            dimension is first
        """

        super().__init__()

        if src_embed == 'mlp':
            src_embed_layer = embeddings.ProbabilityEmbedding(d_model)
        elif src_embed == 'matrix_scale':
            src_embed_layer = embeddings.MatrixScaleEmbedding(
                d_model, source_size)
        elif src_embed == 'spectra_continuous':
            src_embed_layer = embeddings.NMRContinuousEmbedding(d_model)
        elif src_embed == 'nn.embed':
            src_embed_layer = nn.Embedding(
                source_size, d_model, padding_idx=src_pad_token)
        elif src_embed is None:
            src_embed_layer = None
        else:
            raise ValueError("Unsupported source embedding")

        assert (tgt_embed == 'nn.embed')
        tgt_embed_layer = nn.Embedding(
            target_size, d_model, padding_idx=tgt_pad_token)

        src_forward_function = getattr(forward_fxns, src_forward_function)
        tgt_forward_function = getattr(forward_fxns, tgt_forward_function)

        self.network = transformer.Transformer(src_embed_layer, tgt_embed_layer,
                                               src_pad_token, tgt_pad_token,
                                               src_forward_function, tgt_forward_function,
                                               d_model, nhead, num_encoder_layers, num_decoder_layers,
                                               dim_feedforward, dropout, activation, custom_encoder,
                                               custom_decoder, target_size, source_size,
                                               layer_norm_eps, batch_first, norm_first, device, dtype)
        self.initialize_weights()
        self.freeze_components = freeze_components
        self.device = device
        self.dtype = dtype

    def initialize_weights(self) -> None:
        """Initializes network weights
        Non-1D parameters are initialized using Xavier initialization
        """
        for p in self.network.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def freeze(self) -> None:
        """Disables gradients for specific components of the network

        Args:
            components: A list of strings corresponding to the model components
                to freeze, e.g. src_embed, tgt_embed.
        """
        # TODO: This will need careful testing
        if self.freeze_components is not None:
            for component in self.freeze_components:
                if hasattr(self.network, component):
                    for param in getattr(self.network, component).parameters():
                        param.requires_grad = False

    def forward(self,
                x: Tuple[Tensor, Tuple],
                y: Tuple[Tensor, Tensor]) -> Tensor:
        return self.network(x, y)

    def get_loss(self,
                 x: Tuple[Tensor, Tuple],
                 y: Tuple[Tensor],
                 loss_fn: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
        return self.network.get_loss(x, y, loss_fn)
