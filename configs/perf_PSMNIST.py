import snntorch as SNNTORCH_LIB
from snntorch import surrogate as SNNTORCH_SURROGATE

from spikingjelly.activation_based import neuron, surrogate
from src.utils import Triangle

class Config():
    
    ### Dataset ###
    
    dataset = 'PSMNIST'
    datasets_path = 'Datasets/PSMNIST'
    seed = 0
    time_window = 784
    
    ### General ###
    
    epochs = 200
    batch_size = 32
    
    bias = True
    use_batch_norm = False
    
    results_dir = ''
    
    num_workers = 8
    
    ### Model architechture ###
    
    hidden_layers = [64, 212, 212] 
    
    input_size = 1
    output_size = 10
    
    recurrent_dropout_rate = 0.2 
    feedforward_dropout_rate = 0.1
    
    init_ff_weights = 'default' # 'kaiming' or 'normal' or 'default'
    init_dcls_weights = 'default' # 'kaiming' or 'normal' or 'default'
    
    ### Spiking neuron configuration ###
    
    # --- SpikingJelly Config ---
    neuron_module = neuron.LIFNode
    backend = 'torch'
    tau = 2.0
    v_threshold = 1.0
    v_reset = None # None if soft reset
    surrogate_function = Triangle.apply
    
    # --- snnTorch Config ---
    beta = 0.60653  # This is exp(-1/tau) = exp(-1/2.0)
    surrogate_function_snntorch = Triangle.apply # Use your custom surrogate
    reset_mechanism_snntorch = 'subtract' # Matches v_reset = None
    
    # --- Common Config ---
    detach_reset = False
    decay_input=False
    step_mode = 'm'
    store_v_seq = True
    
    ### Recurrent delays ###
    
    rec_kernel_size = 3

    use_sig_p = True
    
    init_rec_delay = 'half_normal' # 'uniform' or 'half_normal'
    # if 'uniform':
    max_rec_delay = 50 
    # if 'half_normal':
    delay_std_init = 12 
    
    rec_delay_init_gain = 1.0
    
    sigma_init = 10.0
    sigma_decay = 0.95
    
    ### Feedforward delays ###
    
    DCLSversion = 'v1' 
    
    kernel_count = 1
    max_feedforward_delay = 30
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay%2==1 else max_feedforward_delay+1 
    
    left_padding = max_feedforward_delay - 1
    right_padding = 0
    
    init_pos_a = -max_feedforward_delay//2
    init_pos_b = max_feedforward_delay//2
    
    ### Optimization ###
    
    optim = 'adamW' # 'adam' or 'adamw'
    
    scheduler_weights = 'onecycle' # 'cos' or 'onecycle'
    scheduler_pos = 'onecycle' # 'onecycle' or 'cos'
    
    lr_w = 1.25e-4
    lr_positions = 6.25e-3
    
    weight_decay = 1e-2