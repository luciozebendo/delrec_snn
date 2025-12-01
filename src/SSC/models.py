import torch
import matplotlib.pyplot as plt
import wandb
import math

from spikingjelly.activation_based import layer
from DCLS.construct.modules import Dcls1d

# custom methods
from src.recurrent_neurons import SNNTorchAxonalRecDel, ConvSNNTorchAxonalRecDel

class dcls_module(Dcls1d):
    """
    A wrapper for the Dcls1d (learnable delay convolution) layer.
    This class handles the necessary data shape permutations to make
    the SNN's standard (T, B, N) tensor compatible with the
    convolution's (B, C, T) format.
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
        assert x.dim() == 3 # (Time, Batch, Neurons)
        x = x.permute(1,2,0) # change it for 1D convolution
        x = torch.nn.functional.pad(x, (self.left_padding, self.right_padding), 'constant', 0) # apply temporal padding
        x = super().forward(x) # apply the learnable delay convolution
        x = x.permute(2,0,1) # permute back to (Time, Batch, Neurons)
        return x
        
class modified_batchnorm(layer.BatchNorm1d):
    """
    A wrapper for SpikingJelly's BatchNorm1d to handle (T, B, N) input.
    The base SpikingJelly layer expects an extra dimension, so this
    module unsqueezes and squeezes to make them compatible.
    """
    def __init__(self, num_features, step_mode='m'):
        super().__init__(num_features, step_mode=step_mode)
        
    def forward(self, x):
        assert x.dim() == 3
        # add a dummy dimension (T, B, N, 1) for compatibility, apply BN, then remove it
        return super().forward(x.unsqueeze(3)).squeeze() 

class spike_registrator(torch.nn.Module):
    """
    A "spy" module that acts as an identity function but
    saves a clone of the spikes that pass through it.
    This is used to calculate the spike regularization cost.
    """
    def __init__(self):
        super().__init__()
        self.spikes = None # variable to store the passing spikes

    def forward(self, x):
        assert x.dim() == 3
        self.spikes = x.clone()
        return x # pass the spikes through unchanged

    def reset(self):
        self.spikes = []

class SNN(torch.nn.Module):
    """
    A base Spiking Neural Network (SNN) class.
    """
    def __init__(self, config):
        super().__init__()
        
        assert config.dataset in ['SSC', 'PSMNIST'], "This SNN is designed for SSC or PSMINST datasets."
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        # Build the hidden layers
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
        
        # Add the final output layer
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def forward(self, x): 
        assert x.dim() == 3
        x = self.layers(x) # pass the input through all layers sequentially
        return x
        
    def log_params(self):
        """Logs mean weights and max gradients for standard Linear layers."""
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
        """Initializes weights for all Linear layers based on the config."""
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
        """Helper function to enforce constraints on all recurrent delay parameters."""
        for m in self.layers:
            if isinstance(m, SNNTorchAxonalRecDel) or isinstance(m, ConvSNNTorchAxonalRecDel):
                m.clamp_recurrent_delays() # keep delays within a valid range
                
    def round_pos(self):
        """Rounds the learned recurrent delays to the nearest integer for inference."""
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, SNNTorchAxonalRecDel) or isinstance(m, ConvSNNTorchAxonalRecDel):
                    m.recurrent_delays.round_()
                    m.clamp_recurrent_delays()
        
    def forward(self, x):
        return super().forward(x)
    
    def log_params(self):
        """
        Extends the base log_params to add detailed logging for
        the 'SNNTorchAxonalRecDel' layers, including delay histograms.
        """
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
        