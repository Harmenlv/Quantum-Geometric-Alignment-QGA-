import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

from federated_utils import (
    get_datasets,
    dirichlet_non_iid_partition,
    evaluate_model,
)

from hetero_models import ARCH_REGISTRY, ResNet18_CIFAR
from hetero_aggregators import (
    compute_quantum_fidelity_weights,
    distill_server_model,
    evaluate_ensemble,
)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 基础配置 =====================
DATASET_NAME = "cifar10"
NUM_CLIENTS = 20
ALPHA = 0.1
GLOBAL_ROUNDS = 100
LOCAL_EPOCHS = 2
LR = 0.01
LR_DECAY = 0.98
EVAL_INTERVAL = 5

CALIB_SAMPLES = 5000
CALIB_BATCH_SIZE = 128

SEEDS = [42]

NUM_WORKERS = 8
PIN_MEMORY = True

RESULT_DIR = "results/hetero"
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULT_DIR, "round_logs")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ===================== 断点 / 早停 / 日志 =====================
ENABLE_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5

ENABLE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 20

ENABLE_ROUND_LOG = True
PRINT_INTERVAL = 10

# ===================== 异构配置 =====================
ARCH_PRESETS = {
    "homo": ["resnet18"] * 20,
    "mild": ["resnet18"] * 10 + ["mobilenetv2"] * 10,
    "cross": ["resnet18"] * 5 + ["vgg11"] * 5 + ["mobilenetv2"] * 5 + ["simplecnn"] * 5,
}

METHODS = ["ensemble", "feddf", "conf_feddf"]

TARGET_ACC = 70.0


# ===================== 本地构造 calibration dataset =====================
def build_calibration_dataset(dataset_name, num_samples=5000, seed=42, transform=None):
    np.random.seed(seed)

    if dataset_name == "cifar10":
        base_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform,
        )
    elif dataset_name == "cifar100":
        base_dataset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform,
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    total = len(base_dataset)
    num_samples = min(num_samples, total)
    indices = np.random.choice(total, num_samples, replace=False)
    return Subset(base_dataset, indices)


def build_distill_loader(dataset_name, num_samples=5000, seed=42, batch_size=128):
    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    distill_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    calib_dataset = build_calibration_dataset(
        dataset_name=dataset_name,
        num_samples=num_samples,
        seed=seed,
        transform=distill_transform,
    )

    calib_loader = DataLoader(
        calib_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )
    return calib_loader


# ===================== 客户端定义 =====================
class HeteroClient:
    def __init__(self, client_id, dataset, model_name, num_classes, device, lr=0.01, local_epochs=2):
        self.client_id = client_id
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        self.model_name = model_name

        self.model = ARCH_REGISTRY[model_name](num_classes=num_classes).to(device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    def local_train(self, round_idx=0, lr_decay=0.98):
        self.model.train()
        current_lr = self.lr * (lr_decay ** round_idx)
        for g in self.optimizer.param_groups:
            g["lr"] = current_lr

        total_loss = 0.0
        total_seen = 0

        for _ in range(self.local_epochs):
            for inputs, labels in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_seen += bs

        return total_loss / max(total_seen, 1)


def make_clients(client_datasets, arch_list, num_classes):
    clients = []
    for i in range(len(client_datasets)):
        clients.append(
            HeteroClient(
                client_id=i,
                dataset=client_datasets[i],
                model_name=arch_list[i],
                num_classes=num_classes,
                device=DEVICE,
                lr=LR,
                local_epochs=LOCAL_EPOCHS,
            )
        )
    return clients


def estimate_logit_comm_MB(num_clients, num_samples, num_classes, dtype_bytes=4):
    total_bytes = num_clients * num_samples * num_classes * dtype_bytes
    return total_bytes / (1024 * 1024)


# ===================== 断点保存 / 加载 =====================
def save_checkpoint(exp_name, round_idx, server_state, client_states,
                    acc_history, comm_total, best_acc, best_round,
                    patience_counter):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    torch.save({
        "round_idx": round_idx,
        "server_state": server_state,
        "client_states": client_states,
        "acc_history": acc_history,
        "comm_total": comm_total,
        "best_acc": best_acc,
        "best_round": best_round,
        "patience_counter": patience_counter,
    }, ckpt_path)


def load_checkpoint(exp_name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"检测到断点，从第 {ckpt['round_idx'] + 1} 轮继续: {exp_name}")
    return ckpt


# ===================== 单组实验 =====================
def run_one_setting(seed, arch_key, method):
    exp_name = f"hetero_{DATASET_NAME}_alpha{ALPHA}_{arch_key}_{method}_seed{seed}"

    train_set, test_set, num_classes = get_datasets(DATASET_NAME)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    client_datasets = dirichlet_non_iid_partition(train_set, NUM_CLIENTS, ALPHA, seed=seed)
    calib_loader = build_distill_loader(
        dataset_name=DATASET_NAME,
        num_samples=CALIB_SAMPLES,
        seed=42,
        batch_size=CALIB_BATCH_SIZE,
    )

    arch_list = ARCH_PRESETS[arch_key]
    clients = make_clients(client_datasets, arch_list, num_classes)

    server_model = ResNet18_CIFAR(num_classes=num_classes).to(DEVICE)

    client_weights = np.array([c.num_samples for c in clients], dtype=np.float64)
    client_weights = client_weights / client_weights.sum()

    start_round = 0
    acc_history = []
    comm_total = 0.0
    best_acc = 0.0
    best_round = 0
    patience_counter = 0
    fidelity_values = []
    client_mix_weights = client_weights.tolist()
    early_stop_triggered = False
    last_test_acc = 0.0

    comm_per_round_MB = estimate_logit_comm_MB(
        num_clients=NUM_CLIENTS,
        num_samples=CALIB_SAMPLES,
        num_classes=num_classes,
    )

    # 加载断点
    if ENABLE_CHECKPOINT:
        ckpt = load_checkpoint(exp_name)
        if ckpt is not None:
            start_round = min(ckpt["round_idx"] + 1, GLOBAL_ROUNDS)
            if method in ["feddf", "conf_feddf"]:
                server_model.load_state_dict(ckpt["server_state"], strict=False)
            for i, cs in enumerate(ckpt["client_states"]):
                clients[i].model.load_state_dict(cs, strict=False)

            acc_history = ckpt["acc_history"]
            comm_total = ckpt["comm_total"]
            best_acc = ckpt["best_acc"]
            best_round = ckpt["best_round"]
            patience_counter = ckpt["patience_counter"]
            last_test_acc = acc_history[-1] if acc_history else 0.0

    desc_str = f"{arch_key}/{method}/s{seed}"
    pbar = tqdm(
        range(start_round, GLOBAL_ROUNDS),
        desc=desc_str,
        initial=start_round,
        total=GLOBAL_ROUNDS,
        mininterval=1.0,
    )

    for r in pbar:
        # --- 1) 本地训练 ---
        for client in clients:
            client.local_train(round_idx=r, lr_decay=LR_DECAY)

        # --- 2) 客户端权重 ---
        conf_teacher_acc = 0.0
        ref_teacher_acc = 0.0
        use_fidelity = False

        if method == "conf_feddf":
            client_mix_weights, fidelity_values, conf_teacher_acc, ref_teacher_acc, gate_fell_back, is_active = (
                compute_quantum_fidelity_weights(
                    [c.model for c in clients],
                    calib_loader,
                    DEVICE,
                    reference_weights=client_weights,
                    round_idx=r,
                    warmup_rounds=10,
                    old_weights=client_mix_weights if r > 0 else None,
                    ema_momentum=0.95,
                    uniform_blend=0.30,
                    qgfl_bias=0.35,
                    qgfl_scale=0.65,
                )
            )
            use_fidelity = is_active and not gate_fell_back
        else:
            client_mix_weights = client_weights.tolist()

        # --- 3) 服务器蒸馏 ---
        if method in ["feddf", "conf_feddf"]:
            server_model = distill_server_model(
                server_model=server_model,
                client_models=[c.model for c in clients],
                calib_loader=calib_loader,
                device=DEVICE,
                client_weights=client_mix_weights,
                temperature=5.0,
                distill_epochs=20,
                lr=0.01,
                hard_weight=0.1,
                ema_decay=0.95,
            )

        # --- 4) 通信量累计 ---
        comm_total += comm_per_round_MB

        # --- 5) 评估 ---
        is_eval_round = (r == 0) or ((r + 1) % EVAL_INTERVAL == 0) or (r == GLOBAL_ROUNDS - 1)

        if is_eval_round:
            if method == "ensemble":
                test_acc = evaluate_ensemble(
                    [c.model for c in clients],
                    test_loader,
                    DEVICE,
                    client_weights=client_weights.tolist(),
                )
            else:
                test_acc = evaluate_model(server_model, test_loader, DEVICE)

            if ENABLE_EARLY_STOP:
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_round = r + 1
                    patience_counter = 0
                    if ENABLE_CHECKPOINT:
                        save_checkpoint(
                            exp_name + "_best",
                            r,
                            server_model.state_dict(),
                            [c.model.state_dict() for c in clients],
                            acc_history,
                            comm_total,
                            best_acc,
                            best_round,
                            patience_counter,
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        early_stop_triggered = True
        else:
            test_acc = last_test_acc

        acc_history.append(test_acc)
        last_test_acc = test_acc

        # --- 6) 进度条 ---
        postfix = {
            "Acc": f"{test_acc:.2f}%",
            "Best": f"{best_acc:.2f}%",
            "Pat": f"{EARLY_STOP_PATIENCE - patience_counter}",
            "Comm": f"{comm_total:.1f}MB",
        }
        if method == "conf_feddf":
            avg_fid = float(np.mean(fidelity_values)) if fidelity_values else 0.0
            postfix["Fid"] = f"{avg_fid:.3f}"
            postfix["W/Ref"] = f"{conf_teacher_acc:.1f}/{ref_teacher_acc:.1f}"
            postfix["门控"] = "保真度" if use_fidelity else "回退FedDF"

        pbar.set_postfix(postfix, refresh=True)

        if early_stop_triggered:
            postfix["状态"] = "已早停"
            pbar.set_postfix(postfix, refresh=True)
            pbar.close()
            break

        # --- 7) 周期打印 ---
        if (r + 1) % PRINT_INTERVAL == 0:
            print(
                f"\n[轮次 {r + 1}] {arch_key} | {method} | seed={seed} | "
                f"test_acc={test_acc:.2f}% | best_acc={best_acc:.2f}% | "
                f"耐心={EARLY_STOP_PATIENCE - patience_counter} | comm={comm_total:.2f}MB"
            )
            if method == "conf_feddf":
                avg_fid = float(np.mean(fidelity_values)) if fidelity_values else 0.0
                print(
                    f"   [QGFL] 平均保真度={avg_fid:.3f} | "
                    f"加权teacher={conf_teacher_acc:.2f}% | "
                    f"数据量teacher={ref_teacher_acc:.2f}% | "
                    f"门控={'保真度' if use_fidelity else '回退FedDF'}"
                )

        # --- 8) 断点保存 ---
        if ENABLE_CHECKPOINT and (r + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                exp_name,
                r,
                server_model.state_dict(),
                [c.model.state_dict() for c in clients],
                acc_history,
                comm_total,
                best_acc,
                best_round,
                patience_counter,
            )

        # --- 9) 每轮日志 ---
        if ENABLE_ROUND_LOG:
            log_row = {
                "round": r + 1,
                "top1_acc": test_acc,
                "comm_MB": comm_total,
                "best_acc": best_acc,
                "is_eval": is_eval_round,
                "avg_quantum_fidelity": float(np.mean(fidelity_values)) if fidelity_values else 0.0,
                "conf_teacher_acc": conf_teacher_acc,
                "ref_teacher_acc": ref_teacher_acc,
                "use_fidelity": use_fidelity,
            }
            log_path = os.path.join(LOG_DIR, f"{exp_name}_log.csv")
            pd.DataFrame([log_row]).to_csv(
                log_path,
                mode="a",
                header=not os.path.exists(log_path),
                index=False,
            )

    # 实验结束，清理普通断点
    if ENABLE_CHECKPOINT:
        normal_ckpt = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
        if os.path.exists(normal_ckpt):
            os.remove(normal_ckpt)

    final_acc = acc_history[-1] if acc_history else 0.0
    rounds_to_target = GLOBAL_ROUNDS
    for idx, acc in enumerate(acc_history):
        if acc >= TARGET_ACC:
            rounds_to_target = idx + 1
            break

    num_params = sum(p.numel() for p in server_model.parameters()) / 1e6

    return {
        "seed": seed,
        "arch": arch_key,
        "method": method,
        "final_acc": final_acc,
        "best_acc": best_acc,
        "best_round": best_round,
        "rounds_to_target": rounds_to_target,
        "comm_MB": comm_total,
        "params_M": num_params,
        "early_stopped": early_stop_triggered,
    }


# ===================== 主入口 =====================
if __name__ == "__main__":
    all_rows = []
    total_runs = len(SEEDS) * len(ARCH_PRESETS) * len(METHODS)
    print(f"\n开始组B异构实验，共 {total_runs} 组子实验\n")

    for seed in SEEDS:
        for arch_key in ARCH_PRESETS.keys():
            for method in METHODS:
                row = run_one_setting(seed, arch_key, method)
                all_rows.append(row)
                print(
                    f"\n完成: seed={seed} | arch={arch_key} | method={method} | "
                    f"best={row['best_acc']:.2f}% | final={row['final_acc']:.2f}%\n"
                )
                print("-" * 90)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(RESULT_DIR, "hetero_results.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("\n组B异构实验完成")
    print(f"结果已保存: {out_csv}")
    print(df)