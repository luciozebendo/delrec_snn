import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
import math

from DCLS.construct.modules import Dcls1d
from src.recurrent_neurons import SNNTorchAxonalRecDel, ConvSNNTorchAxonalRecDel

class dcls_module(Dcls1d):
    """
    (This class is copied from your SHD snn.py)
    """
    def __init__(
        self,
        config,
        in_channels,
        out_channels,
        groups,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_count=config.kernel_count,
            groups=groups,
            dilated_kernel_size=config.max_feedforward_delay,
            bias=config.bias,
            version=config.DCLSversion,
            )
        
        self.config = config
        self.left_padding, self.right_padding = config.left_padding, config.right_padding
        
    def forward(self, x):
        assert x.dim() == 3 # (T, B, N)
        x = x.permute(1,2,0) # (batch, neurons, time)
        x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding), 'constant', 0)
        x = super().forward(x) # (batch, neurons, time)
        x = x.permute(2,0,1) # (time, batch, neurons)
        return x
        
class modified_batchnorm(nn.Module):
    """
    (This class is copied from your SHD snn.py)
    """
    def __init__(self, num_features, step_mode='m'):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features=num_features)
        
    def forward(self, x):
        assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
        T, B, N = x.shape
        x_reshaped = x.reshape(T * B, N)
        x_bn = self.bn(x_reshaped)
        return x_bn.reshape(T, B, N)

class spike_registrator(torch.nn.Module):
    """
    (This class is copied from your SHD snn.py)
    """
    def __init__(self):
        super().__init__()
        self.spikes = None

    def forward(self, x):
        assert x.dim() == 3
        self.spikes = x.clone()
        return x

    def reset(self):
        self.spikes = []

class SNN(torch.nn.Module):
    """
    Base SNN class adapted for PSMNIST.
    """
    def __init__(self, config):
        super().__init__()
        
        assert config.dataset == 'PSMNIST', "This SNN is designed for PSMNIST dataset."
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim
            
            layers.append(torch.nn.Dropout(config.feedforward_dropout_rate))
            
            layers.append(config.neuron_module(
                tau = config.tau,
                decay_input = config.decay_input,
                v_reset = config.v_reset,
                v_threshold = config.v_threshold,
                surrogate_function = config.surrogate_function,
                detach_reset = config.detach_reset,
                step_mode = config.step_mode,
                backend = config.backend,
                store_v_seq = config.store_v_seq,
                )
            )
                
            layers.append(spike_registrator())
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
        
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def forward(self, x): 
        assert x.dim() == 3
        x = self.layers(x)
        # sum spikes/voltages over time for classification
        x = torch.sum(x, 0) 
        return x
        
    def log_params(self):
        logs = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Linear):
                w = torch.abs(layer.weight).mean()
                w_grad_max = layer.weight.grad.abs().max().item() if layer.weight.grad is not None else 0.0
                logs.update({
                        f'w_{idx}': w,
                        f'w_grad_max_{idx}': w_grad_max,
                    })
        wandb.log(logs)
        
    def init_weights(self):
        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif self.config.init_ff_weights == 'normal':
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
                elif self.config.init_ff_weights == 'default':
                    pass
    
class SNN_recurrent_delays(SNN):
    def __init__(self, config):
        super().__init__(config) 
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim
            
            layers.append(torch.nn.Dropout(config.feedforward_dropout_rate))
            
            layers.append(ConvSNNTorchAxonalRecDel(config, layer_dim))
                
            layers.append(spike_registrator())
                
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
                
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
            
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def clamp_delays(self):
        for m in self.layers:
            if isinstance(m, SNNTorchAxonalRecDel) or isinstance(m, ConvSNNTorchAxonalRecDel):
                m.clamp_recurrent_delays()
                
    def round_pos(self):
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, SNNTorchAxonalRecDel) or isinstance(m, ConvSNNTorchAxonalRecDel):
                    m.recurrent_delays.round_()
                    m.clamp_recurrent_delays()
        
    def forward(self, x):
        return SNN.forward(self, x) 
    
    def log_params(self):
        super().log_params()
        
        logs = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, SNNTorchAxonalRecDel) or isinstance(layer, ConvSNNTorchAxonalRecDel):
                    logs[f'sigma_rec{idx}'] = layer.sigma
                    curr_pos_rec = layer.recurrent_delays.cpu().detach().numpy()
                    logs[f'pos_rec{idx}'] = curr_pos_rec.mean()
                    
                    fig, ax = plt.subplots()
                    ax.hist(curr_pos_rec.reshape(-1), bins=20)
                    ax.set_title(f'Recurrent Delays Distribution Block {idx}')
                    logs[f'pos_rec_hist_plot{idx}'] = wandb.Image(fig)
                    plt.close(fig)
                    
                    if hasattr(layer, 'recurrent_kernel'):
                        rec_w = layer.recurrent_kernel
                    else:
                        rec_w = layer.recurrent_weights
                    rec_w_mean = torch.abs(rec_w).mean()
                    rec_w_grad_max = rec_w.grad.abs().max().item() if rec_w.grad is not None else 0.0

                    logs.update({
                        f'recurrent_w_{idx}': rec_w_mean,
                        f'recurrent_w_grad_max_{idx}': rec_w_grad_max,
                    })
            
                    rec_d = layer.recurrent_delays
                    rec_d_grad_max = rec_d.grad.abs().max().item() if rec_d.grad is not None else 0.0
                    logs[f'recurrent_delay_grad_max_{idx}'] = rec_d_grad_max
                    
                    if layer.use_sig_p:
                        logs[f"p_spread_mean_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().mean().item()
                        logs[f"p_spread_std_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().std().item()
                    
        wandb.log(logs)