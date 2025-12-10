import numpy as np
import os
from tqdm import tqdm
from src.datasets import BinnedSpikingHeidelbergDigits

for split in ['train', 'test']:
    dataset = BinnedSpikingHeidelbergDigits('Datasets/SHD', train=(split=='train'), n_bins=5, duration_ms=1.0)
    
    save_dir = f'Datasets/SHD/preprocessed_{split}'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in tqdm(range(len(dataset))):
        frames, label = dataset[i]
        np.savez_compressed(f'{save_dir}/{i}.npz', frames=frames, label=label)