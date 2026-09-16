# Quantum Geometric Alignment Framework (QGAF)

<p align="center">
  <strong>Quantum Geometric Alignment for Heterogeneous Neural Operators</strong>
</p>

<p align="center">
  Official implementation of QGAF and its federated extension QGFL.
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-green" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange" alt="PyTorch">
  </a>
  <img src="https://img.shields.io/badge/GPU-RTX%204090%2024GB-purple" alt="GPU">
  <img src="https://img.shields.io/badge/Paper-Under%20Review-blue" alt="Paper">
</p>

---

## 📌 Overview

**Quantum Geometric Alignment Framework (QGAF)** is a quantum-geometric framework for measuring and aligning heterogeneous neural operators.

The central idea is to represent neural operators through **density-matrix / Choi-state representations**, and then quantify their functional similarity using quantum-geometric metrics rather than directly comparing their parameter coordinates.

Based on this representation, we develop **QGFL**, a federated learning extension that incorporates geometric similarity into client aggregation, with additional support for heterogeneous client architectures through server-side knowledge distillation.

---

## ✨ Highlights

* **Quantum-geometric operator representation**

  * Neural operators are mapped to normalized density-matrix representations.
  * The leading eigenvector is used to obtain a compact quantum pure-state representation.

* **Coordinate-free functional similarity**

  * Quantum fidelity provides a similarity measure between operator representations.
  * The Fubini–Study distance provides the corresponding geometric distance.

* **Permutation-aware representation**

  * The representation is designed to reduce sensitivity to neuron ordering and parameter-coordinate permutations.

* **QGFL aggregation**

  * Client contributions are weighted according to geometric similarity.
  * Adaptive blending and exponential moving average (EMA) are incorporated for stable aggregation.

* **Heterogeneous federated learning**

  * Supports clients with different neural architectures.
  * Uses logit-level geometric alignment and server-side knowledge distillation.

* **Comprehensive evaluation**

  * CIFAR-10 and CIFAR-100.
  * Multiple Dirichlet non-IID settings.
  * Multiple random seeds.
  * Standard federated learning baselines.
  * Heterogeneous architecture experiments.
  * Ablation studies.

---

## 🧠 Method Overview

The QGAF framework consists of three main stages:

```text
Neural Operator
      │
      ▼
Operator Response Extraction
      │
      ▼
Covariance / Density-Matrix Representation
      │
      ▼
Quantum State Representation
      │
      ├───────────────┐
      ▼               ▼
Fidelity F       Fubini–Study Distance
      │               │
      └───────┬───────┘
              ▼
      Geometric Alignment
              │
              ▼
      QGFL Aggregation
              │
              ▼
       Global Model
```

## 🧠 Method Overview

### 1. Operator Representation

For a neural operator, forward responses are collected using PyTorch forward hooks.

The response covariance matrix is normalized to construct a density-matrix representation:

$\rho = C / \mathrm{Tr}(C)$.

The dominant eigenvector of the density matrix is obtained from the eigenvalue equation:

$\rho\psi = \lambda_{\max}\psi$,

where $\lambda_{\max}$ denotes the largest eigenvalue. The corresponding eigenvector $\psi$ is used as a compact quantum-state representation of the neural operator.

### 2. Quantum Fidelity

For two normalized operator states $\psi_1$ and $\psi_2$, their quantum fidelity is defined as:

$F(\psi_1,\psi_2) = |\langle\psi_1 \mid \psi_2\rangle|^2$.

A larger fidelity value indicates stronger geometric alignment between the two operator representations.

### 3. Fubini–Study Distance

The corresponding Fubini–Study distance is defined as:

$d_{\mathrm{FS}}(\psi_1,\psi_2) = \arccos\left(\sqrt{F(\psi_1,\psi_2)}\right)$.

This metric measures the geometric separation between the two normalized quantum-state representations in the underlying projective Hilbert space.
---

# 🔬 QGFL

**QGFL** extends the geometric representation to federated learning.

Instead of treating all client models as points in a common parameter-coordinate system, QGFL evaluates their functional representations and uses geometric alignment to determine aggregation weights.

A simplified aggregation procedure is:

```text
Client Models
     │
     ├── Client 1 ──► Operator Representation
     ├── Client 2 ──► Operator Representation
     ├── Client 3 ──► Operator Representation
     │
     ▼
Quantum-Geometric Similarity
     │
     ▼
Fidelity-Based Client Weights
     │
     ▼
Adaptive Blending
     │
     ▼
EMA Smoothing
     │
     ▼
Global Aggregation
```

The implementation includes:

* fidelity-based client weighting;
* adaptive aggregation;
* weight smoothing;
* exponential moving average (EMA);
* standard federated baselines;
* heterogeneous architecture aggregation.

---

# 🏗️ Heterogeneous Federated Learning

QGAF also provides a heterogeneous federated learning extension for clients with different neural architectures.

The model zoo currently includes:

| Architecture        | Description                       |
| ------------------- | --------------------------------- |
| `SimpleCNN`         | Lightweight convolutional network |
| `VGG11_CIFAR`       | VGG-based CIFAR model             |
| `MobileNetV2_CIFAR` | MobileNetV2-based CIFAR model     |
| `ResNet18_CIFAR`    | ResNet18-based CIFAR model        |

For cross-architecture settings, direct parameter averaging is generally not applicable.

QGAF therefore uses geometric information at the representation/logit level and combines it with server-side knowledge distillation.

---

# 📂 Repository Structure

```text
QGAF/
├── README.md
├── LICENSE
├── requirements.txt
│
├── quantum_metric.py
│   └── Core quantum-geometric metric
│       ├── operator response extraction
│       ├── covariance construction
│       ├── density-matrix representation
│       ├── quantum fidelity
│       └── Fubini–Study distance
│
├── aggregators.py
│   └── Federated aggregation methods
│       ├── FedAvg
│       ├── FedProx
│       ├── FedNova
│       ├── FedDyn
│       ├── FedDF
│       ├── SCAFFOLD
│       └── QGFL
│
├── hetero_aggregators.py
│   └── Heterogeneous federated learning
│       ├── geometric fidelity weighting
│       ├── server-side distillation
│       └── ensemble evaluation
│
├── federated_utils.py
│   └── Dataset and training utilities
│       ├── CIFAR-10 / CIFAR-100
│       ├── Dirichlet partitioning
│       ├── client wrapper
│       ├── evaluation
│       ├── communication cost estimation
│       └── BatchNorm calibration
│
├── hetero_models.py
│   └── Heterogeneous model zoo
│       ├── SimpleCNN
│       ├── VGG11_CIFAR
│       ├── MobileNetV2_CIFAR
│       └── ResNet18_CIFAR
│
├── pruning_methods.py
│   └── Pruning baselines
│       ├── magnitude pruning
│       └── random pruning
│
├── federated_main.py
│   └── Main federated learning experiments
│
├── federated_main_hetero.py
│   └── Heterogeneous federated learning experiments
│
└── federated_ablation_resnet18.py
    └── Ablation studies
```

---

# ⚙️ Requirements

* Python 3.10+
* PyTorch ≥ 2.0
* CUDA-enabled GPU recommended

The main dependencies include:

```text
torch
torchvision
numpy
pandas
tqdm
scikit-learn
matplotlib
seaborn
scipy
PyYAML
tensorboard
einops
networkx
joblib
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd QGAF
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Quick Start

## 1. Main Federated Learning Experiments

The main experiment compares QGFL with conventional federated learning algorithms:

* FedAvg
* FedProx
* FedNova
* FedDyn
* FedDF
* SCAFFOLD
* QGFL

Run:

```bash
python federated_main.py
```

The experiments support:

* CIFAR-10;
* CIFAR-100;
* multiple Dirichlet \(\alpha\) values;
* multiple random seeds;
* configurable client numbers;
* configurable local epochs;
* configurable global communication rounds.

---

## 2. Heterogeneous Federated Learning

To evaluate cross-architecture federated learning:

```bash
python federated_main_hetero.py
```

This experiment uses heterogeneous client architectures defined in:

```text
hetero_models.py
```

and aggregation logic implemented in:

```text
hetero_aggregators.py
```

---

## 3. Ablation Study

To isolate the contributions of the core geometric components:

```bash
python federated_ablation_resnet18.py
```

The ablation experiments are designed to investigate the effects of:

* quantum-state / operator alignment;
* fidelity-based weighting;
* adaptive aggregation;
* smoothing mechanisms.

---

# 🧪 Experiments

The repository is organized around three experimental settings.

### A. Homogeneous Federated Learning

All clients use the same model architecture.

```text
CIFAR-10 / CIFAR-100
        │
        ▼
 Dirichlet Non-IID
        │
        ▼
 Homogeneous Clients
        │
        ▼
FedAvg / FedProx / ...
        │
        ▼
       QGFL
```

### B. Heterogeneous Federated Learning

Clients use different architectures:

```text
Client 1 → SimpleCNN
Client 2 → VGG11
Client 3 → MobileNetV2
Client 4 → ResNet18
```

The server performs geometric alignment and knowledge distillation instead of directly averaging incompatible model parameters.

### C. Ablation

Ablation experiments isolate individual components of QGFL to determine their contribution to the overall behavior.

---

# 📊 Results

Experimental results are automatically stored under:

```text
results/
```

Typical outputs include:

```text
results/
├── csv/
├── xlsx/
├── logs/
├── checkpoints/
└── figures/
```

Depending on the experiment configuration, the output files may contain:

* global test accuracy;
* training loss;
* client statistics;
* aggregation weights;
* fidelity measurements;
* communication cost;
* convergence curves;
* heterogeneous FL evaluation results.

---

# 🔁 Reproducibility

The main experiment scripts expose the principal experimental parameters near the beginning of each script.

Typical parameters include:

```python
NUM_CLIENTS
LOCAL_EPOCHS
GLOBAL_ROUNDS
LEARNING_RATE
DIRICHLET_ALPHA
SEEDS
BATCH_SIZE
```

For reproducible experiments, we recommend fixing:

1. random seeds;
2. dataset partition;
3. model initialization;
4. optimizer configuration;
5. federated communication rounds;
6. local training epochs.

---

# 💻 Hardware

The implementation is designed for CUDA-enabled PyTorch environments.

The main experiments were validated on a system equipped with:

```text
GPU: NVIDIA RTX 4090
VRAM: 24 GB
```

A GPU with sufficient VRAM is recommended for the heterogeneous and ablation experiments.

Actual memory consumption depends on:

* batch size;
* number of clients;
* model architecture;
* local training configuration;
* number of stored operator responses.

---

# 📚 Citation

If you find QGAF useful in your research, please consider citing the associated manuscript:

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

> **Paper status:** Revised Manuscript Submitted

---

# 👨‍🔬 Academic Homepage

For publications, citations, and research updates, please visit the author's Google Scholar profile:

[Google Scholar](https://scholar.google.com/citations?user=d3mvChQAAAAJ&hl=en)

---

# 📄 License

The repository license will be specified before public release.

```text
TODO: MIT / Apache-2.0
```

---

# 🌐 Project Status

| Component                     | Status        |
| ----------------------------- | ------------- |
| QGAF core metric              | Available     |
| Quantum-state representation  | Available     |
| Fidelity calculation          | Available     |
| Fubini–Study distance         | Available     |
| QGFL aggregation              | Available     |
| Homogeneous FL experiments    | Available     |
| Heterogeneous FL experiments  | Available     |
| Ablation studies              | Available     |
| CIFAR-10                      | Supported     |
| CIFAR-100                     | Supported     |
| Reproducibility configuration | Available     |
| Public paper link             | To be updated |
| Final license                 | To be updated |

---

<p align="center">
  <sub>Quantum Geometric Alignment Framework (QGAF)</sub>
</p>

<p align="center">
  <sub>Geometry • Quantum Representation • Federated Learning • Heterogeneous Neural Operators</sub>
</p>

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


