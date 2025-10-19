import torch
import matplotlib.pyplot as plt
import wandb
import math

from spikingjelly.activation_based import layer
from DCLS.construct.modules import Dcls1d

# custom methods
from src.recurrent_neurons import axonal_recdel

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
        # initialize the Dcls1d convolution with preconfigured parameters
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
        self.left_padding, self.right_padding = config.left_padding, config.right_padding # Get padding values
        
    def forward(self, x):
        assert x.dim() == 3 # (Time, Batch, Neurons)
        x = x.permute(1,2,0) # change to (Batch, Neurons, Time) for 1D convolution
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
    It builds a simple feedforward SNN with linear layers,
    spiking neurons, dropout, and optional batch norm.
    """
    def __init__(self, config):
        super().__init__()
        
        assert config.dataset in ['SSC', 'PSMNIST'], "This SNN is designed for SSC or PSMINST datasets."
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        # this build the hidden layers
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias)) # feedforward linear layer
            dim_buffer = layer_dim # update the dimension for the next layer
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
            
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
                
            layers.append(spike_registrator()) # add the function to record spikes
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
        
        # add the final output layer
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        
        self.layers = torch.nn.Sequential(*layers) # unite all layers into a sequential module
        
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
    """
    An SNN that inherits from the base SNN but replaces the standard
    spiking neurons with 'axonal_recdel' modules.
    This introduces learnable recurrent (feedback) delays.
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        # build the hidden layers, but this time with the recurrent delay module
        for idx, layer_dim in enumerate(config.hidden_layers):
            layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
            
            # use axonal_recdel instead of the standard neuron module
            layers.append(axonal_recdel(config, layer_dim, config.neuron_module))
                
            layers.append(spike_registrator())
                
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
                
        layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
            
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def clamp_delays(self):
        """Helper function to enforce constraints on all recurrent delay parameters."""
        for m in self.layers:
            if isinstance(m, axonal_recdel):
                m.clamp_recurrent_delays() # keep delays within a valid range
                
    def round_pos(self):
        """Rounds the learned recurrent delays to the nearest integer for inference."""
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, axonal_recdel):
                    m.recurrent_delays.round_()
                    m.clamp_recurrent_delays()
        
    def forward(self, x):
        return super().forward(x) # use the same foward method as the base SNN
    
    def log_params(self):
        """
        Extends the base log_params to add detailed logging for
        the 'axonal_recdel' layers, including delay histograms.
        """
        super().log_params()
        
        logs = {}
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, axonal_recdel):
                    
                    logs[f'sigma_rec{idx}'] = layer.sigma
                    curr_pos_rec = layer.recurrent_delays.cpu().detach().numpy()
                    logs[f'pos_rec{idx}'] = curr_pos_rec.mean()
                    
                    # create and log a histogram of the delay distribution
                    fig, ax = plt.subplots()
                    ax.hist(curr_pos_rec.reshape(-1), bins=20)
                    ax.set_title(f'Recurrent Delays Distribution Block {idx}')
                    logs[f'pos_rec_hist_plot{idx}'] = wandb.Image(fig)
                    plt.close(fig)
                    
                    # store recurrent weight and gradient information
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
                    
                    # if ture, log spread parameters
                    if layer.use_sig_p:
                        logs[f"p_spread_mean_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().mean().item()
                        logs[f"p_spread_std_{idx}"] = (2 * torch.sigmoid(layer.p_spread) * layer.sigma).detach().std().item()
                    
        wandb.log(logs)
        
class SNN_vanilla_recurrent(SNN_recurrent_delays):
    """
    A control model for ablation studies.
    It inherits from SNN_recurrent_delays but sets all
    recurrent delays to 0 and makes them non-trainable.
    This creates a standard SNN with 1-step recurrence.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # --- Find all recurrent delay layers ---
        for layer in self.layers:
            if isinstance(layer, axonal_recdel):
                # --- Overwrite delays with a non-trainable zero tensor ---
                recurrent_delays = layer.recurrent_delays
                layer.recurrent_delays = torch.nn.Parameter(torch.zeros_like(recurrent_delays), requires_grad=False)
                
    def forward(self, x):
        return super().forward(x)
    
class SNN_fixed_recurrent_delays(SNN_recurrent_delays):
    """
    Another control model. It inherits from SNN_recurrent_delays
    but freezes the recurrent delays (requires_grad=False).
    They are initialized but do not learn.
    """
    def __init__(self, config):
        super().__init__(config)
        
        for layer in self.layers:
            if isinstance(layer, axonal_recdel):
                layer.recurrent_delays.requires_grad = False # Freeze delays
                self.sigma = 0.
                self.sigma_init = 0.
                
    def forward(self, x):
        return super().forward(x)
    
class SNN_feedforward_delays(SNN):
    """
    An SNN that inherits from the base SNN but replaces the standard
    'torch.nn.Linear' layers with the 'dcls_module'.
    This introduces learnable feedforward (convolutional) delays.
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        dim_buffer = config.input_size
        
        # --- Re-build the layers, this time with the feedforward delay module ---
        for idx, layer_dim in enumerate(config.hidden_layers):
            
            # --- KEY DIFFERENCE: Use dcls_module instead of torch.nn.Linear ---
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = layer_dim,
                        groups = 1,
                    )
                    )
            dim_buffer = layer_dim
            
            # --- The rest of the block is standard (dropout, neuron, spy, batchnorm) ---
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))
                
            layers.append(config.neuron_module(
            tau = config.tau,
            # ... (rest of neuron parameters) ...
            store_v_seq = config.store_v_seq,
            )
                           )
            
            layers.append(spike_registrator())
            
            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))
                
        # --- Add the final output layer (either Linear or dcls_module) ---
        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = config.output_size,
                        groups = 1,
                    ))
            
        self.layers = torch.nn.Sequential(*layers)
        
        self.init_weights()
        
    def forward(self, x):
        return super().forward(x)
    
    def init_weights(self):
        """Extends base init_weights to also initialize dcls_module weights and delays."""
        for m in self.layers:
            # --- Initialize feedforward Linear layers (if any) ---
            if isinstance(m, torch.nn.Linear):
                if self.config.init_ff_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                # ... (other init methods) ...
                elif self.config.init_ff_weights == 'default':
                    pass
                    
            # --- Initialize feedforward dcls_module weights ---
            if isinstance(m, dcls_module):
                if self.config.init_dcls_weights == 'kaiming':
                    torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                # ... (other init methods) ...
                elif self.config.init_dcls_weights == 'default':
                    torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) # Match nn.Linear default
            
            # --- Initialize feedforward dcls_module delays (P) ---
            if isinstance(m, dcls_module):
                torch.nn.init.uniform_(m.P, a = self.config.init_pos_a, b = self.config.init_pos_b)
                m.clamp_parameters() # Ensure initial delays are valid
                
    def clamp_delays(self, train=True):
        """Helper function to enforce constraints on all feedforward delay parameters."""
        for m in self.layers:
            if isinstance(m, dcls_module):
                m.clamp_parameters()
                        
    def round_pos(self):
        """Rounds the learned feedforward delays (P) to the nearest integer for inference."""
        with torch.no_grad():
            for m in self.layers:
                if isinstance(m, dcls_module):
                    m.P.round_()
                    m.clamp_parameters()
                        
    def log_params(self):
        """
        Extends the base log_params to add detailed logging for
        the 'dcls_module' layers, including delay histograms.
        """
        super().log_params() # Log parameters for Linear layers first
        
        logs = {}
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, dcls_module):
                
                # --- Log mean feedforward delay ---
                curr_pos_ff = layer.P.cpu().detach().numpy()
                logs[f'pos_feedforward{idx}'] = curr_pos_ff.mean()
                
                # --- Create and log a histogram of the delay distribution ---
                fig, ax = plt.subplots()
                ax.hist(curr_pos_ff.reshape(-1), bins=20)
                ax.set_title(f'Feedforward Delays Distribution Block {idx}')
                logs[f'pos_feedforward_hist_plot{idx}'] = wandb.Image(fig)
                plt.close(fig)
                
                # --- Log feedforward weights and delay gradients ---
                ff_w = layer.weight
                ff_w_mean = torch.abs(ff_w).mean()
                ff_w_grad_max = ff_w.grad.abs().max().item() if ff_w.grad is not None else 0.0
                ff_d_grad_max = layer.P.grad.abs().max().item() if layer.P.grad is not None else 0.0

                logs.update({
                    f'feedforward_w_{idx}': ff_w_mean,
                    f'feedforward_w_grad_max_{idx}': ff_w_grad_max,
                    f'recurrent_delay_grad_max_{idx}': ff_d_grad_max,
                })
                
        wandb.log(logs)
        
class SNN_recurrent_and_feedforward_delays(SNN_feedforward_delays, SNN_recurrent_delays):
    """
    The "all-in" model: combines both learnable feedforward and
    learnable recurrent delays using multiple inheritance.
    """

    def __init__(self, config):
        # --- Initialize the base SNN class ---
        super(SNN_feedforward_delays, self).__init__(config)  

        self.config = config

        layers = []
        dim_buffer = config.input_size

        # --- Build layers combining both delay types ---
        for idx, layer_dim in enumerate(config.hidden_layers):
            # --- 1. Add feedforward delay layer (dcls_module) ---
            if config.no_delay_in_first_layer and idx == 0:
                layers.append(torch.nn.Linear(dim_buffer, layer_dim, bias=config.bias))
            else:
                layers.append(
                    dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = layer_dim,
                        groups = 1,
                    )
                    )
            dim_buffer = layer_dim
            
            layers.append(layer.Dropout(config.feedforward_dropout_rate, step_mode='m'))

            # --- 2. Add recurrent delay layer (axonal_recdel) ---
            layers.append(axonal_recdel(config, layer_dim, config.neuron_module))
                
            layers.append(spike_registrator())

            if config.use_batch_norm:
                layers.append(modified_batchnorm(layer_dim, step_mode='m'))

        # --- Add final output layer (with or without delays) ---
        if config.no_delay_in_last_layer:
            layers.append(torch.nn.Linear(dim_buffer, config.output_size, bias=config.bias))
        else:
                layers.append(dcls_module(
                        config,
                        in_channels = dim_buffer,
                        out_channels = config.output_size,
                        groups = 1,
                    ))

        self.layers = torch.nn.Sequential(*layers)

        # --- Initialize weights for both feedforward and recurrent types ---
        SNN_feedforward_delays.init_weights(self)

    def forward(self, x):
        # --- Use the original SNN.forward() to pass data sequentially ---
        return SNN.forward(self, x)
    
    def clamp_delays(self):
        """Dispatcher: clamps delays for BOTH feedforward and recurrent layers."""
        SNN_feedforward_delays.clamp_delays(self)
        SNN_recurrent_delays.clamp_delays(self)
        
    def round_pos(self):
        """Dispatcher: rounds delays for BOTH feedforward and recurrent layers."""
        SNN_feedforward_delays.round_pos(self)
        SNN_recurrent_delays.round_pos(self)
        
    def log_params(self):
        """Dispatcher: logs parameters for BOTH feedforward and recurrent layers."""
        SSNN_feedforward_delays.log_params(self)
        SNN_recurrent_delays.log_params(self)