import torch
import torch.nn.functional as F
import wandb

# custom methods
from src.recurrent_neurons import axonal_recdel
from src.SSC.snn import dcls_module, modified_batchnorm
from src.utils import *

def get_spike_cost(model, normalize="NT"):
    """
    Calculates a regularization term to penalize high firing rates in the SNN.
    This encourages sparsity, which is often desired in spiking models for efficiency.
    It iterates through the model, finds spike recordings, and computes their mean squared value.
    """
    costs = []
    # get all spike recordings saved in the model's modules
    for m in model.modules():
        if m.__class__.__name__ == "spike_registrator":
            spk = getattr(m, "spikes", None)
            if spk is None or not torch.is_tensor(spk):
                continue
            spk = spk.float() 

            # compute the squared L2 norm of the spikes
            if spk.dim() != 3: # this is a fix for cases where dimensions may be squeezed
                costs.append(0.5 * spk.pow(2).mean())
                continue

            # normalize the cost by time steps and number of neurons
            T, B, N = spk.shape
            if normalize == "NT":
                per_sample = 0.5 * spk.pow(2).sum(dim=(0, 2)) / (T * N)
                costs.append(per_sample.mean())
            else:
                costs.append(0.5 * spk.pow(2).mean())

    # return the average cost across all layers, or zero in the case no spikes were recorded
    if not costs:
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
    return torch.stack(costs).mean()

def train(train_loader, model, optimizer, epoch, device, config, penalize_spikes=False):
    """
    Executes one full training epoch.
    It iterates over the training dataset, performs forward and backward passes,
    updates model parameters, and logs the performance.
    """
    train_loss = 0
    correct    = 0
    total      = 0

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # move to the correct device and permute dimensions for SNNs (time, batch and neurons), making time the first dimension
        inputs = inputs.permute(1,0,2).float().to(device)
        targets = targets.to(device)
        
        # forward pass: reset recurrent states, then pass data through the model
        reset_states(model=model)
        outputs = model(inputs)
        loss = calc_loss_SSC(outputs, targets) 
        
        # if true, add the spike regularization penalty to the main loss
        if penalize_spikes:
            spike_cost = get_spike_cost(model)
            loss += config.spike_penalty * spike_cost
            wandb.log({"spike_cost": spike_cost.item()})

        # sum loss and accuracy values for the epoch 
        train_loss += loss.item()
        correct += calc_metric_SSC(outputs, targets) 
        total += targets.size(0)

        # backward pass and optimizer step
        for opt in optimizer: opt.zero_grad() # clear gradients from the previous step.
        loss.backward() # compute updated gradients again
        for opt in optimizer: opt.step() # update model parameters.

        # print a progress bar with updated metrics
        progress_bar(
            batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total)
            )
            
    # calculate and write in the log the average loss and accuracy for the entire epoch
    avg_loss = train_loss / len(train_loader)
    avg_acc = 100. * correct / total
    wandb.log({
        'train/loss': avg_loss,
        'train/acc': avg_acc,
        'epoch': epoch,
    })

    model.log_params()
            
    return avg_acc, avg_loss

def test(test_loader, model, epoch, device, config, penalize_spikes=False):
    """
    Evaluates the model on a test or validation dataset.
    This function is similar to train() but without gradient calculations or parameter updates.
    """
    test_loss = 0
    correct = 0
    total = 0

    model.eval()
    
    # if the model has learnable delays, round them to the nearest integer
    if hasattr(model, 'round_pos'):
        model.round_pos()
        
    with torch.no_grad(): # disable gradient computation to speed up the process
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            
            # run the forward pass
            inputs = inputs.permute(1,0,2).float().to(device)
            targets = targets.to(device)
            reset_states(model=model)
            outputs = model(inputs)
            loss = calc_loss_SSC(outputs, targets)
            
            # if true again, add the spike regularization cost
            if penalize_spikes:
                spike_cost = get_spike_cost(model)
                loss += config.spike_penalty * spike_cost

            # update progress bar 
            test_loss += loss.item()
            correct += calc_metric_SSC(outputs, targets) 
            total += targets.size(0)
            progress_bar(
                batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total)
            )
    
    avg_loss = test_loss / len(test_loader)
    avg_acc = 100. * correct / total
    wandb.log({
        'test/loss': avg_loss,
        'test/acc': avg_acc,
        'epoch': epoch,
    })
    
    return avg_acc, avg_loss

def init_optim_sche(model, config):
    """
    Initializes optimizers and learning rate schedulers.
    It separates model parameters into different groups (e.g., weights vs. delays)
    to allow for different learning rates and optimization strategies for each group.
    """
    weights_norm = [] # parameters for normalization layers.
    weights = [] # standard connection weights.
    positions = [] # learnable delay parameters.

    # assign parameters to each layer groups
    for m in model.layers:
        if isinstance(m, torch.nn.Linear):
            weights.append(m.weight)
            if config.bias: weights.append(m.bias)
                
        elif isinstance(m, axonal_recdel): # layer with recurrent weights and delays
            weights.append(m.recurrent_weights)
            positions.append(m.recurrent_delays)
            if hasattr(m, 'p_spread'): positions.append(m.p_spread)
            
        elif isinstance(m, dcls_module): # layer with feedforward weights and delays
            weights.append(m.weight)
            if config.bias: weights.append(m.bias)
            positions.append(m.P)
            
        elif isinstance(m, modified_batchnorm): # norm layer
            weights_norm.append(m.weight)
            if config.bias: weights_norm.append(m.bias)

    optimizer = []
    scheduler = []

    # add one optimizer for weights and another for delays
    if config.optim == 'adam':
        # optimizer for standard weights and normalization parameters
        optimizer.append(torch.optim.Adam([{'params':weights, 'lr':config.lr_w, 'weight_decay':config.weight_decay},
                                            {'params':weights_norm, 'lr':config.lr_w, 'weight_decay':0},]))
        # optimizer for delay parameters
        optimizer.append(torch.optim.Adam([{'params':positions, 'lr':config.lr_positions, 'weight_decay':0}]))
    else:
        raise NotImplementedError

    # lr scheduler for the weights optimizer
    if config.scheduler_weights == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[0], T_max=config.epochs))
    elif config.scheduler_weights == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[0], max_lr=config.lr_w, total_steps=config.epochs))
    else:
        raise NotImplementedError
    
    # lr scheduler for the delays optimizer
    if config.scheduler_pos == 'cos':
        scheduler.append(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer[1], T_max=config.epochs))
    elif config.scheduler_pos == 'onecycle':
        scheduler.append(torch.optim.lr_scheduler.OneCycleLR(optimizer[1], max_lr=config.lr_positions, total_steps=config.epochs))
    else:
        raise NotImplementedError
    
    return optimizer, scheduler