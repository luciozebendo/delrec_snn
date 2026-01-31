import torch 
import numpy as np
import h5py
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split 
from torch.distributions.binomial import Binomial
from typing import Callable, Optional

from spikingjelly.datasets.shd import SpikingHeidelbergDigits
from spikingjelly.datasets import pad_sequence_collate 

from src.utils import seed_everything

def load_dataset(config):
    """Selects and loads the appropriate dataset based on configuration."""
    if config.dataset == 'SHD':
        return SHD_dataloaders(config)
    elif config.dataset == 'SSC':
        return SSC_dataloaders(config)
    elif config.dataset == 'PSMNIST':
        return PS_MNIST_dataloaders(config)
    else:
        raise ValueError(f"Dataset {config.dataset} is not supported.")

def SHD_dataloaders(config):
    """Initializes DataLoaders for the Spiking Heidelberg Digits dataset."""
    seed_everything(config.seed, is_cuda=True)
    train_dataset = PreprocessedSHD(config.datasets_path, train=True)
    test_dataset = PreprocessedSHD(config.datasets_path, train=False)
    # split into train/validation
    valid_dataset = test_dataset
  
    if config.use_augmentations:
        train_dataset = SHDTripleAugDataset(train_dataset, shift_max=config.shift_max, thin_p=config.thin_p, jitter_in_blend=config.jitter_in_blend)

    # pad_sequence_collate is used to handle variable length sequences
    train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=0)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader

class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    """Custom dataset class for SHD that applies spatial binning."""
    def __init__(self, root: str, n_bins: int, duration_ms: float = 5.0, **kwargs):
        # Force data_type='event' to get raw spike data
        kwargs['data_type'] = 'event'
        super().__init__(root, **kwargs)
        self.n_bins = n_bins
        self.duration_ms = duration_ms  # Temporal bin width in milliseconds
        
    def __getitem__(self, i: int):
        # Get raw event data
        events = {'t': self.h5_file['spikes']['times'][i], 
                  'x': self.h5_file['spikes']['units'][i]}
        label = self.h5_file['labels'][i]
        
        # Get spike times (in seconds) and neuron indices
        spike_times = np.array(events['t'])  # in seconds
        spike_neurons = np.array(events['x'])  # neuron indices 0-699
        
        # Determine time range
        if len(spike_times) > 0:
            t_max = spike_times.max()
        else:
            t_max = 1.0  # default 1 second
            
        # Create time bins (convert to milliseconds)
        n_time_bins = int(np.ceil(t_max * 1000 / self.duration_ms))
        
        # Create frame representation
        frames = np.zeros((n_time_bins, 700), dtype=np.float32)
        
        # Fill in spikes
        for t, n in zip(spike_times, spike_neurons):
            time_bin = int(t * 1000 / self.duration_ms)  # Convert to ms and bin
            if time_bin < n_time_bins and n < 700:
                frames[time_bin, int(n)] += 1
        
        # Spatial binning: 700 -> 140 neurons
        N_binned = 700 // self.n_bins
        binned_frames = np.zeros((n_time_bins, N_binned), dtype=np.float32)
        
        for j in range(N_binned):
            binned_frames[:, j] = frames[:, self.n_bins*j : self.n_bins*(j+1)].sum(axis=1)
        
        return binned_frames, label

class SHDTripleAugDataset(Dataset):
    """Dataset wrapper that triples data with jitter, thinning, and blending augmentations."""
    def __init__(self, base_ds_or_subset, shift_max: int = 40, thin_p: float = 0.5, jitter_in_blend: bool = False):
        super().__init__()
        self.base = base_ds_or_subset
        self.shift_max = int(shift_max)
        self.thin_p = float(thin_p)
        self.jitter_in_blend = bool(jitter_in_blend)

        self.is_subset = hasattr(self.base, "indices") and hasattr(self.base, "dataset")
        self.indices = list(self.base.indices) if self.is_subset else list(range(len(self.base)))
        self.n = len(self.indices)

        self.labels = np.empty(self.n, dtype=np.int64)
        for pos in range(self.n):
            _, y = self._get_item_pos(pos)
            self.labels[pos] = int(y)

        self.class_pos = {}
        for pos, y in enumerate(self.labels):
            self.class_pos.setdefault(int(y), []).append(pos)

    def _get_item_pos(self, pos: int):
        return self.base[pos] if self.is_subset else self.base[self.indices[pos]]

    @staticmethod
    def _time_shift_per_neuron(x: torch.Tensor, shift_max: int) -> torch.Tensor:
        """
        Applies a different time shift to each neuron independently.
        x shape: (Time, Neurons)
        """
        if shift_max <= 0:
            return x
        T, N = x.shape
        out = x.new_zeros(T, N)
        shifts = torch.randint(-shift_max, shift_max + 1, (N,), device=x.device)
        for n in range(N):
            s = int(shifts[n])
            if s > 0:
                out[s:, n] = x[:-s, n]
            elif s < 0:
                out[:s, n] = x[-s:, n]
            else:
                out[:, n] = x[:, n]
        return out

    @staticmethod
    def _com_time(x: torch.Tensor) -> float:
        T = x.shape[0]
        mass_t = x.sum(dim=1)
        denom = mass_t.sum()
        if denom <= 0:
            return 0.5 * (T - 1)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        return float((t * mass_t).sum() / denom)

    @staticmethod
    def _thin_binomial(x: torch.Tensor, p: float) -> torch.Tensor:
        xi = x.clamp_min(0).round()
        if xi.numel() == 0:
            return xi
        dist = Binomial(total_count=xi, probs=torch.tensor(p, device=xi.device))
        return dist.sample()

    @staticmethod
    def _align_and_pad_pair(x: torch.Tensor, xb: torch.Tensor, align: int):
        T1, N = x.shape
        T2, N2 = xb.shape
        assert N == N2, "Neuron dimension mismatch"
        pad_x_left = max(0, -align)
        pad_xb_left = max(0, align)
        T_out = max(T1 + pad_x_left, T2 + pad_xb_left)
        x_pad = x.new_zeros(T_out, N)
        xb_pad = xb.new_zeros(T_out, N)
        x_pad[pad_x_left: pad_x_left + T1] = x
        xb_pad[pad_xb_left: pad_xb_left + T2] = xb
        return x_pad, xb_pad

    def _sample_same_class_partner(self, y: int, avoid_pos: int):
        pool = self.class_pos.get(int(y), [])
        if len(pool) <= 1:
            return None
        j = avoid_pos
        while j == avoid_pos:
            j = pool[np.random.randint(0, len(pool))]
        return j if int(self.labels[j]) == int(y) else None

    def __len__(self):
        return 3 * self.n

    def __getitem__(self, idx: int):
        if idx < self.n:
            x, y = self._get_item_pos(idx)
            return torch.as_tensor(x, dtype=torch.float32), int(y)

        if idx < 2 * self.n:
            pos = idx - self.n
            x, y = self._get_item_pos(pos)
            x = torch.as_tensor(x, dtype=torch.float32)
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        pos = idx - 2 * self.n
        x, y = self._get_item_pos(pos)
        x = torch.as_tensor(x, dtype=torch.float32)
        partner_pos = self._sample_same_class_partner(int(y), pos)
        if partner_pos is None:
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        xb, yb = self._get_item_pos(partner_pos)
        xb = torch.as_tensor(xb, dtype=torch.float32)
        if int(yb) != int(y):
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        align = int(round(self._com_time(x) - self._com_time(xb)))
        x, xb = self._align_and_pad_pair(x, xb, align)
        if self.jitter_in_blend and self.shift_max > 0:
            x = self._time_shift_per_neuron(x, self.shift_max)
            xb = self._time_shift_per_neuron(xb, self.shift_max)
        x = self._thin_binomial(x, self.thin_p) + self._thin_binomial(xb, self.thin_p)
        return x, int(y)

def SSC_dataloaders(config):
    """Initializes the Spiking Speech Commands dataset with efficient HDF5 iterators."""
    seed_everything(config.seed, is_cuda=True)
    T = config.time_window
    max_time = 1.4

    root_path = config.datasets_path
    dataset = config.dataset

    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    valid_file = h5py.File(os.path.join(root_path, dataset.lower()+'_valid.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train, y_train = train_file['spikes'], train_file['labels']
    x_valid, y_valid = valid_file['spikes'], valid_file['labels']
    x_test, y_test = test_file['spikes'], test_file['labels']

    train_loader = SSC_SpikeIterator(x_train, y_train, config.batch_size, T, 700, max_time, n_bins=config.n_bins, shuffle=True)
    valid_loader = SSC_SpikeIterator(x_valid, y_valid, config.batch_size, T, 700, max_time, n_bins=config.n_bins, shuffle=True)
    test_loader = SSC_SpikeIterator(x_test, y_test, config.batch_size, T, 700, max_time, n_bins=config.n_bins, shuffle=False)

    return train_loader, valid_loader, test_loader

class SSC_SpikeIterator:
    """Custom iterator to efficiently load SSC HDF5 spike data in batches."""
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, n_bins=1, shuffle=True, device='cuda:0', indices=None, label_map=None):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.float32)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))

        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)

        self.n_bins = n_bins
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.number_of_batches)

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[self.batch_size*self.counter:min(self.batch_size*(self.counter+1), self.num_samples)]

            # construct sparse tensor representation of spikes
            coo = [[] for _ in range(3)]
            for bc, idx in enumerate(batch_index):
                times = np.digitize(self.firing_times[idx], self.time_bins)
                units = self.units_fired[idx]
                batch = [bc]*len(times)
                coo[0].extend(batch)
                coo[1].extend(times)
                coo[2].extend(units)

            i = torch.LongTensor(coo).to(self.device)
            v = torch.FloatTensor(np.ones(len(coo[0]))).to(self.device)

            X_batch = torch.sparse.FloatTensor(i, v, torch.Size([len(batch_index), self.nb_steps, self.nb_units])).to_dense().to(self.device)

            # suggest vectorizing this binning loop for GPU efficiency.
            binned_len = X_batch.shape[-1] // self.n_bins
            binned_frames = torch.zeros((len(batch_index), self.nb_steps, binned_len)).to(self.device)
            for i in range(binned_len):
                binned_frames[:,:,i] = X_batch[:,:,self.n_bins*i:self.n_bins*(i+1)].sum(axis=-1)

            y_batch = torch.tensor(self.labels_[batch_index], device=self.device).long()
            X_batch = binned_frames

            self.counter += 1
            return X_batch, y_batch
        else:
            raise StopIteration

def PS_MNIST_dataloaders(config):
    """Loads standard MNIST dataset for Permuted Sequential MNIST tasks."""
    seed_everything(config.seed, is_cuda=True)
    config.time_window = 784
    config.input_dim = 1
    config.output_dim = 10

    is_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if is_cuda else {}

    dataset_train = datasets.MNIST(config.datasets_path, train=True, download=True, transform=transforms.ToTensor())

    train_size = int(0.83 * len(dataset_train))
    val_size = len(dataset_train) - train_size
    dataset_train, dataset_val = random_split(dataset_train, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed))

    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(datasets.MNIST(config.datasets_path, train=False, transform=transforms.ToTensor()), batch_size=config.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader

class PreprocessedSHD(Dataset):
    """Dataset that loads preprocessed SHD .npz files."""
    def __init__(self, root: str, train: bool = True):
        super().__init__()
        self.split = 'train' if train else 'test'
        self.data_dir = os.path.join(root, f'preprocessed_{self.split}')
        
        # Get all .npz files
        self.files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.npz')])
        print(f"Loaded {len(self.files)} preprocessed {self.split} samples from {self.data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load preprocessed file
        data = np.load(os.path.join(self.data_dir, self.files[idx]))
        frames = data['frames'].astype(np.float32)  # (T, 140)
        label = int(data['label'])
        return frames, label