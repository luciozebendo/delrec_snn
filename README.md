# Convolutional Recurrence with Learnable Delays in Spiking Neural Networks

Code for **"Combining Convolution and Delay Learning in Recurrent Spiking Neural Networks"**
L. F. S. Zebendo, E. Cicciarella, M. Rossi — *EUSIPCO 2026*

📄 [Paper (arXiv:2604.15997)](https://arxiv.org/abs/2604.15997)

---

## What this does

Recurrent SNNs with learnable axonal delays ([DelRec](https://arxiv.org/abs/2501.07331)) use a dense recurrent weight matrix, costing `N²` parameters per layer. For temporal signals with strong local structure — audio spectrograms, where adjacent cochlear channels are correlated — that connectivity is largely redundant.

This work replaces the dense recurrent matrix with a **1D convolution** (kernel size `k = 3`) along the neuron dimension, while keeping the learnable delay mechanism intact. Each neuron receives recurrent input only from itself and its two immediate neighbours.

The recurrent parameter count drops from `N² + N` to `k + N`.

## Results

| Dataset | Method | Layers | Test Acc. [%] | Rec. Params | Inference [ms] |
|---------|--------|-------:|--------------:|------------:|---------------:|
| SHD | DelRec | 4 | 90.41 ± 0.83 | 196,608 | 38.03 |
| SHD | **Ours** | 4 | **91.51 ± 0.70** | **9** | **1.51** |
| SSC | DelRec | 3 | 82.58 ± 0.08 | 196,608 | 112.64 |
| SSC | **Ours** | 3 | 78.59 ± 0.39 | **9** | **4.19** |

Accuracies are mean ± std over 9 seeds (SHD) and 3 seeds (SSC).

On SHD the convolutional variant matches DelRec accuracy with a **52× faster inference** and four orders of magnitude fewer recurrent parameters. On SSC it trails by ~4 points, using the baseline hyperparameters from the original paper without additional tuning.

An ablation replacing learnable delays with fixed ones (`d = 1`, or fixed at the learned mean/median) costs more than 5 percentage points on SHD, confirming that the delays — not the convolution alone — carry the temporal modelling.

## Installation

```bash
git clone https://github.com/luciozebendo/delrec-snn.git
cd delrec-snn
pip install -r requirements.txt
```

Requirements pin `torch==2.3.0+cu118`, so install PyTorch from the CUDA 11.8 index first:

```bash
pip install torch==2.3.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Tested on Python 3.11.

## Data preparation

Download the [Heidelberg Spiking Datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) (SHD and SSC) and place the HDF5 files under `Datasets/SHD` and `Datasets/SSC`.

SHD requires a one-off preprocessing pass that bins spikes into 140 frequency channels and caches each sample as a compressed `.npz`:

```bash
python pre_process_shd.py
```

This writes `Datasets/SHD/preprocessed_train/` and `Datasets/SHD/preprocessed_test/`. The cached files are not tracked in Git.

## Running

```bash
python run_shd.py              # all seeds sequentially
python run_shd.py --seed 42    # a single seed
python run_ssc.py
```

Model and training hyperparameters live in `configs/` — edit the relevant file rather than passing flags:

| Config | Purpose |
|--------|---------|
| `perf_SHD.py` | Best-performing SHD setup (tuned with Ray Tune + Optuna) |
| `perf_SSC.py` | SSC setup, using DelRec baseline hyperparameters |

Training runs log to [Weights & Biases](https://wandb.ai). Set `WANDB_MODE=offline` to run without an account.

## Repository structure

```
src/
  recurrent_neurons.py   ← convolutional recurrent delay unit (the contribution)
  datasets.py            ← SHD/SSC loading and cochlear binning
  utils.py
  SHD/, SSC/, PSMNIST/   ← per-dataset model definitions and trainers
configs/                 ← hyperparameter configurations
run_*.py                 ← entry points
pre_process_shd.py       ← SHD binning and caching
penalize_spikes.py       ← spike-count regularisation experiments
```

The core modification is in `src/recurrent_neurons.py`: the dense recurrent matrix is replaced by a 1D convolution applied along the neuron dimension, implemented as a 2D convolution with kernel shape `(k, 1)` over the delay buffer, with the delay scheduling mechanism unchanged.

## Citation

```bibtex
@inproceedings{zebendo2026combining,
  title     = {Combining Convolution and Delay Learning in Recurrent Spiking Neural Networks},
  author    = {Zebendo, L{\'u}cio Folly Sanches and Cicciarella, Eleonora and Rossi, Michele},
  booktitle = {Proc. 34th European Signal Processing Conference (EUSIPCO)},
  year      = {2026}
}
```

## Acknowledgements

Built on [DelRec](https://github.com/Thvnvtos/DelRec) (Queant et al.) and the [Heidelberg Spiking Datasets](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/) (Cramer et al.). Developed at the SIGNET group, Department of Information Engineering, University of Padova.
