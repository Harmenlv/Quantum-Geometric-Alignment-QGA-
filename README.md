# QGAF — Quantum Geometric Alignment Framework

**Quantum Geometric Alignment Framework: Heterogeneous Operator Fusion for Knowledge Distillation and Federated Learning**

Official implementation of the quantum-geometric alignment framework (QGAF) and its federated learning extension (QGFL).

This repository contains the **core code** for reproducing the main experiments in the paper. It implements:

- A **quantum-geometric metric** that maps neural operators to density-matrix representations (Choi states) and measures their functional equivalence via quantum fidelity and the Fubini–Study distance — a coordinate-free measure that is invariant to neuron permutation.
- **QGFL**, a federated aggregation method that re-weights client updates by quantum fidelity between client and server operators, with adaptive blending and exponential smoothing, designed for non-IID and heterogeneous federated learning.
- Extensions for **heterogeneous federated learning** (cross-architecture clients), including fidelity-weighted aggregation, server-side knowledge distillation, and ensemble evaluation.
- **Ablation studies** that isolate the contribution of each core component.

---

## Method Overview

| Module | File | Description |
|---|---|---|
| Quantum geometric metric | `quantum_metric.py` | Extracts operator responses via forward hooks, builds normalized covariance (density matrix), takes the leading eigenvector as the quantum pure state `ψ`, and computes quantum fidelity `F = |⟨ψ₁|ψ₂⟩|²` and Fubini–Study distance `d_FS = arccos(√F)`. |
| Federated aggregation | `aggregators.py` | Baseline aggregators (FedAvg, FedProx, FedNova, FedDyn, FedDF, SCAFFOLD) and the core `qgfl_aggregate` with fidelity-based client weighting, weight smoothing, and EMA of fidelity weights. |
| Heterogeneous FL | `hetero_aggregators.py` | Output-level (logits) Choi-state fidelity weighting, server-side distillation (`distill_server_model`), and ensemble evaluation for cross-architecture clients. |
| Data & training utils | `federated_utils.py` | Dataset loading (CIFAR-10/100), Dirichlet non-IID partitioning, `ResNet18_CIFAR`, `FederatedClient`, evaluation, communication-cost estimation, and BatchNorm calibration. |
| Model zoo | `hetero_models.py` | `SimpleCNN`, `VGG11_CIFAR`, `MobileNetV2_CIFAR`, `ResNet18_CIFAR` (registry: `ARCH_REGISTRY`). |
| Pruning baselines | `pruning_methods.py` | Magnitude and random pruning baselines. |

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0 (with CUDA)
- NVIDIA GPU with ≥ 24 GB memory is recommended (code was validated on an RTX 4090)

Install dependencies:

```bash
pip install -r requirements.txt
```

The full dependency list (`requirements.txt`) includes: `torch`, `torchvision`, `numpy`, `pandas`, `tqdm`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `PyYAML`, `tensorboard`, `einops`, `networkx`, `joblib`.

---

## Quick Start

Datasets (CIFAR-10 / CIFAR-100) are downloaded automatically by `torchvision` into `./data` on first run.

**Main comparison experiments** (FedAvg / FedProx / FedNova / FedDyn / FedDF / SCAFFOLD / QGFL, CIFAR-10 & CIFAR-100, multiple Dirichlet `α` and seeds):

```bash
python federated_main.py
```

**Heterogeneous federated learning experiments** (cross-architecture clients):

```bash
python federated_main_hetero.py
```

**Ablation study** (isolating the quantum-state alignment and adaptive weighting modules):

```bash
python federated_ablation_resnet18.py
```

All results (CSV / XLSX tables, logs) are written to the `results/` directory. Experiment configurations (number of clients, local epochs, global rounds, learning rate, `α` values, seeds, etc.) are defined at the top of each script — edit them there before running.

> Note: the `federated_main_hetero.py` script also requires the model definitions in `hetero_models.py` and aggregates via `hetero_aggregators.py`.

---

## Repository Structure

```
QGAF/
├── README.md
├── LICENSE
├── requirements.txt
├── quantum_metric.py              # Core: quantum geometric metric (Choi state, fidelity, Fubini-Study distance)
├── aggregators.py                 # FL aggregators incl. QGFL (fidelity-weighted aggregation)
├── hetero_aggregators.py          # Heterogeneous FL: fidelity weights, server distillation, ensemble eval
├── federated_utils.py             # Data loading, non-IID partition, models, clients, evaluation
├── hetero_models.py               # Model zoo (SimpleCNN / VGG11 / MobileNetV2 / ResNet18)
├── pruning_methods.py             # Pruning baselines
├── federated_main.py              # Main comparison experiments
├── federated_main_hetero.py       # Heterogeneous FL experiments
└── federated_ablation_resnet18.py # Ablation study
```

---

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yourpaper,
  title   = {Quantum Geometric Alignment Framework: Heterogeneous Operator Fusion for Knowledge Distillation and Federated Learning},
  author  = {TODO: authors},
  journal = {TODO: journal},
  year    = {2026},
  note    = {TODO: add DOI / arXiv link}
}
```

---

## License

TODO: add your license (e.g., MIT / Apache-2.0) before public release.
