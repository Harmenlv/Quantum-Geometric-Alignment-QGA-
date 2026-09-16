```markdown
# Quantum Geometric Alignment Framework (QGAF)
[![Paper]([https://img.shields.io/badge/Paper-Under%20Review-blue](https://img.shields.io/badge/Paper-Under%20Review-blue))]([https://TODO.arxiv.link](https://TODO.arxiv.link))
[![Python]([https://img.shields.io/badge/Python-3.10%2B-green](https://img.shields.io/badge/Python-3.10%2B-green))]([https://www.python.org/](https://www.python.org/))
[![PyTorch]([https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange))]([https://pytorch.org/](https://pytorch.org/))
[![GPU]([https://img.shields.io/badge/Recommended-RTX4090%2024GB-purple](https://img.shields.io/badge/Recommended-RTX4090%2024GB-purple))]()
[![License]([https://img.shields.io/badge/License-TODO-lightgrey](https://img.shields.io/badge/License-TODO-lightgrey))]()

> **Quantum Geometric Alignment Framework: Heterogeneous Operator Fusion for Knowledge Distillation and Federated Learning**
> Official implementation of QGAF and its federated extension QGFL.

## 📌 Abstract
Federated learning suffers from performance degradation under **non-IID data and heterogeneous client architectures**, since conventional aggregation (e.g., FedAvg) directly averages model weights without accounting for functional equivalence of different neural operators.

We propose the **Quantum Geometric Alignment Framework (QGAF)**:
1. Map neural network operators into density-matrix representations (Choi states);
2. Use quantum fidelity and Fubini–Study distance as a coordinate-free metric, invariant to neuron permutation;
3. QGFL: fidelity-weighted federated aggregation with adaptive blending and exponential smoothing, tailored for non-IID and cross-architecture heterogeneous federated learning.

This repo contains full code to reproduce main experiments, heterogeneous FL evaluation, ablation studies, and baseline comparisons.

## 📋 Table of Contents
- [Method Overview](#method-overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Citation](#citation)
- [License](#license)

## 🧩 Method Overview
| Module | File | Description |
|---|---|---|
| Quantum geometric metric | `quantum_metric.py` | Extract operator responses via forward hooks, construct normalized covariance (density matrix), take leading eigenvector as quantum pure state $\psi$. Compute quantum fidelity $F = |\langle\psi_1|\psi_2\rangle|^2$ and Fubini–Study distance $d_{\text{FS}} = \arccos(\sqrt{F})$. |
| Federated Aggregation | `aggregators.py` | Baselines (FedAvg, FedProx, FedNova, FedDyn, FedDF, SCAFFOLD) + core `qgfl_aggregate`: fidelity-based client weighting, weight smoothing, EMA for fidelity weights. |
| Heterogeneous FL | `hetero_aggregators.py` | Logit-level Choi-state fidelity weighting, server-side knowledge distillation, ensemble evaluation for cross-architecture clients. |
| Data & Training Utils | `federated_utils.py` | CIFAR-10/100 loading, Dirichlet non-IID partition, ResNet18_CIFAR, client wrapper, evaluation, communication cost estimation, BatchNorm calibration. |
| Model Zoo | `hetero_models.py` | `SimpleCNN`, `VGG11_CIFAR`, `MobileNetV2_CIFAR`, `ResNet18_CIFAR` (registry: `ARCH_REGISTRY`). |
| Pruning Baselines | `pruning_methods.py` | Magnitude pruning & random pruning baselines. |

## 📁 Repository Structure
```
QGAF/
├── README.md
├── LICENSE
├── requirements.txt
├── quantum_metric.py              # Core quantum geometric metric (Choi state, fidelity, Fubini–Study)
├── aggregators.py                 # FL aggregators including QGFL
├── hetero_aggregators.py          # Heterogeneous FL: fidelity weights, server distillation, ensemble eval
├── federated_utils.py             # Data loading, non-IID partition, evaluation, client utilities
├── hetero_models.py               # Model zoo for cross-architecture experiments
├── pruning_methods.py             # Pruning baseline implementations
├── federated_main.py              # Main comparison experiments
├── federated_main_hetero.py       # Heterogeneous federated learning experiments
└── federated_ablation_resnet18.py # Ablation study to isolate core modules
```

## ⚙️ Requirements
- Python 3.10+
- PyTorch ≥ 2.0 (with CUDA)
- **Recommended GPU: NVIDIA RTX 4090 (≥24GB VRAM)** (validated on this hardware)

Install dependencies:
```bash
pip install -r requirements.txt
```
`requirements.txt` includes:
`torch`, `torchvision`, `numpy`, `pandas`, `tqdm`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `PyYAML`, `tensorboard`, `einops`, `networkx`, `joblib`.

## 🚀 Quick Start
Datasets (CIFAR-10 / CIFAR-100) will be automatically downloaded into `./data` on first run via `torchvision`.

### Main Comparison Experiments
> Compare FedAvg / FedProx / FedNova / FedDyn / FedDF / SCAFFOLD / QGFL, on CIFAR-10 & CIFAR-100, multiple Dirichlet $\alpha$ and random seeds.
```bash
python federated_main.py
```

### Heterogeneous Federated Learning
> Cross-architecture clients with server-side distillation
```bash
python federated_main_hetero.py
```

### Ablation Study
> Isolate contributions of quantum-state alignment and adaptive weighting components
```bash
python federated_ablation_resnet18.py
```

All outputs (CSV/XLSX tables, training logs) are saved to `results/`.
Hyperparameters (client count, local epochs, global rounds, lr, Dirichlet $\alpha$, seeds) are configurable at the top of each script.

> Note: `federated_main_hetero.py` depends on model definitions in `hetero_models.py` and aggregation logic in `hetero_aggregators.py`.

## 📝 Citation
If this repository helps your research, please cite our work:
```bibtex
@article{shao2026qgaf,
  title={Quantum Geometric Alignment Framework: Heterogeneous Operator Fusion for Knowledge Distillation and Federated Learning},
  author={Haijian Shao and Xiang Li and Sixun Yan and Jiangyang Tan and Xing Deng and Fei Wang and Yingtao Jiang},
  journal={NEURAL NETWORKS},
  volume={N/A},
  pages={N/A},
  year={2026},
  note={Manuscript No. NEUNET-D-26-00212}
}
```

## 📄 License
> TODO: Add license (MIT / Apache-2.0) before public release.
```
```
## ✨ Highlights
- Propose quantum-geometric metric based on Choi states, invariant to neuron permutation
- QGFL: fidelity-aware aggregation for non-IID federated learning
- Supports heterogeneous cross-architecture clients with server distillation
- Comprehensive baselines + ablation studies on CIFAR-10 / CIFAR-100
```
👨‍🔬 **Academic Homepage**
For publications, citations, and research updates, please visit my Google Scholar profile:
🔗 https://scholar.google.com/citations?user=d3mvChQAAAAJ&hl=en

If you find this project useful in your research, I would greatly appreciate it if you cite my related publications.
Thank you for your support and citations!

❤️ Have a nice day! (English)
🌸 素敵な一日を！ (Japanese)
✨ Bonne journée ! (French)
😊 ¡Que tengas un buen día! (Spanish)
🌿 Einen schönen Tag noch! (German)
☕ 좋은 하루 보내세요! (Korean)
🚀 Желаю хорошего дня! (Russian)
🍃 愿你此行如风，自有繁花相送。 (Chinese)
