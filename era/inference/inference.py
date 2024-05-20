import torch
from torch import nn
import era.inference.inference_fxns as inference_fxns

def run_inference(model: nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  pred_gen_fn: str,
                  pred_gen_opts: dict, 
                  write_freq: int = 100,
                  device: torch.device = None):
    """Run inference on a trained model and generate predictions from the examples in dataset

    Args:
        model: A prepared instance of the trained model
        dataloader: The dataloader to use for inference
        pred_gen_fn: Name of a function that generates predictions from the model, input, and options
        pred_gen_opts: Options to pass to the prediction generator function
    """
    model.eval()
    predictions = []
    pred_gen_fn = getattr(inference_fxns, pred_gen_fn)
    for ibatch, batch in enumerate(dataloader):
        if ibatch > 2:
            break
        if (ibatch % write_freq == 0):
            print(f"On batch {ibatch}")
        batch_prediction = pred_gen_fn(model, batch, pred_gen_opts, device)
        predictions.extend(batch_prediction)
    return predictions
