import snntorch as SNNTORCH_LIB
from snntorch import surrogate as SNNTORCH_SURROGATE
from spikingjelly.activation_based import neuron, surrogate
from src.utils import Triangle, Arctan, ArctanSurrogate

class Config():
    
    ### Dataset ###
    
    dataset = 'SHD'
    datasets_path = 'Datasets/SHD'
    seed = 0
    time_step = 10
    n_bins = 5
    
    ### General ###
    
    epochs = 150
    batch_size = 32
    
    bias = True
    use_batch_norm = True
    
    results_dir = ''
    
    ### Model architechture ###
    #add one more layer here
    hidden_layers = [256, 256] 
    
    input_size = 700 // n_bins
    output_size = 20
    
    recurrent_dropout_rate = 0.2 
    feedforward_dropout_rate = 0.4 
    
    no_delay_in_first_layer = True
    no_delay_in_last_layer = True
    init_ff_weights = 'kaiming'
    
    ### Spiking neuron configuration ###
    
    neuron_module = neuron.LIFNode
    backend = 'torch'
    tau = 1.05 
    v_threshold = 1.0
    v_reset = 0. # None if soft reset
    surrogate_function = surrogate.ATan(alpha = 5.0)
    
    # --- snnTorch Config ---
    neuron_module_snntorch = SNNTORCH_LIB.LIF
    beta = 1.0 - (1.0 / tau) 
    # surrogate_function_snntorch = Triangle.apply
    surrogate_function_snntorch = ArctanSurrogate(alpha=5.0)
    # surrogate_function_snntorch = Arctan.apply
    reset_mechanism_snntorch = 'zero' 
    
    # --- Common ---
    detach_reset = False 
    decay_input=False
    step_mode = 'm'
    store_v_seq = True
    
    ### Recurrent delays ###
    
    rec_kernel_size = 3

    use_sig_p = False
    
    init_rec_delay = 'uniform' # 'uniform' or 'half_normal'
    init_recdel_offset = 10
    max_rec_delay = 30 
    delay_std_init = 15 
    
    rec_delay_init_gain = 1.0 
    
    population_delay_init = 15
    
    sigma_init = 10.0
    sigma_decay = 0.95
    
    ### Feedforward delays ###
    
    DCLSversion = 'v1' 
    
    kernel_count = 1
    max_feedforward_delay = 250//time_step
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay%2==1 else max_feedforward_delay+1 
    
    left_padding = max_feedforward_delay - 1
    right_padding = (max_feedforward_delay-1) // 2
    
    init_pos_a = -max_feedforward_delay//2
    init_pos_b = max_feedforward_delay//2
    
    init_dcls_weights = 'kaiming' # 'kaiming' or 'normal' or 'default'
    
    ### Optimization ###
    
    optim = 'adamW'
    
    scheduler_weights = 'onecycle' # 'cos' or 'onecycle'
    scheduler_pos = 'cos' # 'onecycle' or 'cos'
    
    lr_w = 5e-3 
    lr_positions = 5e-2 
    
    weight_decay = 1e-5
    
    ### Augmentations ###
    
    use_augmentations = True
    
    shift_max = 100
    thin_p = 0.5
    jitter_in_blend = False