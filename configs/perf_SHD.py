import snntorch as SNNTORCH_LIB
from snntorch import surrogate as SNNTORCH_SURROGATE
from spikingjelly.activation_based import neuron, surrogate
from src.utils import Triangle, Arctan, ArctanSurrogate
from src.gated_neurons import LightGRU

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
    hidden_layers = [256, 256, 256, 256]
    input_size = 700 // n_bins
    output_size = 20
    recurrent_dropout_rate = 0.23495685940955754
    feedforward_dropout_rate = 0.4376747992061383
    no_delay_in_first_layer = True
    no_delay_in_last_layer = True
    init_ff_weights = 'kaiming'
    ### Spiking neuron configuration ###
    neuron_module = neuron.LIFNode
    use_light_gru = False
    backend = 'torch'
    tau = 1.1685654746037886
    v_threshold = 1.0
    v_reset = 0.
    surrogate_function = surrogate.ATan(alpha = 5.0)
    # --- snnTorch Config ---
    neuron_module_snntorch = SNNTORCH_LIB.LIF
    beta = 1.0 - (1.0 / tau)
    surrogate_function_snntorch = ArctanSurrogate(alpha=5.0)
    reset_mechanism_snntorch = 'zero'
    # --- Common ---
    detach_reset = False
    decay_input = False
    step_mode = 'm'
    store_v_seq = True
    ### Recurrent delays ###
    rec_kernel_size = 5
    use_sig_p = False
    init_rec_delay = 'uniform'
    init_recdel_offset = 10
    max_rec_delay = 30
    delay_std_init = 15
    rec_delay_init_gain = 1.0
    population_delay_init = 15
    sigma_init = 10.35556186978472
    sigma_decay = 0.9714489091279568
    ### Feedforward delays ###
    DCLSversion = 'v1'
    kernel_count = 1
    max_feedforward_delay = 250 // time_step
    max_feedforward_delay = max_feedforward_delay if max_feedforward_delay % 2 == 1 else max_feedforward_delay + 1
    left_padding = max_feedforward_delay - 1
    right_padding = (max_feedforward_delay - 1) // 2
    init_pos_a = -max_feedforward_delay // 2
    init_pos_b = max_feedforward_delay // 2
    init_dcls_weights = 'kaiming'
    ### Optimization ###
    optim = 'adamW'
    scheduler_weights = 'onecycle'
    scheduler_pos = 'cos'
    lr_w = 0.0013225582163493585
    lr_positions = 0.027880072537707235
    weight_decay = 4.1063106382417676e-06
    ### Augmentations ###
    use_augmentations = True
    shift_max = 100
    ### Early Stopping ###
    patience = 60
    min_epochs = 50
    thin_p = 0.5
    jitter_in_blend = False
