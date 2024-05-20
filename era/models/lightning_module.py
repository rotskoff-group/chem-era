import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import era.models as models
import era.training.loss_fxns as loss_fxns

class LightningModel(L.LightningModule):
    def __init__(self,
                 model_args: dict,
                 training_args: dict):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.build_model_base()
        if self.model_args['load_model'] is not None:
            self.load_model_from_checkpoint(self.model_args['load_model'])
        self.loss_fn = self.build_loss_fn()
        self.save_hyperparameters()
    
    def build_model_base(self):
        model_base = getattr(models, self.model_args['model_type'])
        model_config = self.model_args['model_args']
        self.model = model_base(**model_config)

    def build_loss_fn(self):
        loss_fn_base = getattr(loss_fxns, self.training_args['loss_fn'])
        if self.training_args['loss_fn_args'] is not None:
            loss_fn = loss_fn_base(**self.training_args['loss_fn_args'])
        else:
            loss_fn = loss_fn_base()
        return loss_fn

    def load_model_from_checkpoint(self, filename):
        print(f"Loading model from ckpt file {filename}")
        ckpt = torch.load(filename)['state_dict']
        #Remove the model. prefix from all checkpoint keys
        ckpt = {'.'.join(k.split('.')[1:]) : v for k, v in ckpt.items()}
        curr_state_dict = self.model.state_dict()
        pretrained_dict = {k : v
                           for k, v in ckpt.items()
                           if curr_state_dict[k].shape == v.shape}
        self.model.load_state_dict(pretrained_dict, strict=False)

    def configure_optimizers(self):
        optimizer_base = getattr(optim, self.training_args['optimizer'])
        optimizer = optimizer_base(self.model.parameters(), **self.training_args['optimizer_args'])
        if self.model_args['load_model'] is not None and self.model_args['load_optimizer']:
            ckpt = torch.load(self.model_args['load_model'])['optimizer_state_dict']
            optimizer.load_state_dict(ckpt)
        return optimizer
    
    #def on_validation_model_eval(self, *args, **kwargs):
    #    print("Inside on validation model eval")
    #    super().on_validation_model_eval(*args, **kwargs)
    #    torch.set_grad_enabled(True)

    #def on_test_model_eval(self, *args, **kwargs):
    #    print("Inside on test model eval")
    #    super().on_test_model_eval(*args, **kwargs)
    #    torch.set_grad_enabled(True)

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        loss, _, _ = self.model.get_loss(x, y, self.loss_fn)
        metrics = {f"{prefix}_loss" : loss}
        self.log_dict(metrics, on_step = False, on_epoch = True, sync_dist=True)
        return loss
        
    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            return self._shared_eval(batch, batch_idx, "validation")

    def test_step(self, batch, batch_idx):
        with torch.enable_grad():
            return self._shared_eval(batch, batch_idx, "test")
