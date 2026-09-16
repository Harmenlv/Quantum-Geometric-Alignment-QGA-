import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# 全局开启 cuDNN 自动调优，固定尺寸卷积提速显著
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from federated_utils import (
    get_datasets,
    dirichlet_non_iid_partition,
    build_calibration_loader,
    ResNet18_CIFAR,
    FederatedClient,
    evaluate_model,
    compute_communication_cost,
)
from aggregators import (
    fedavg_aggregate,
    qgfl_aggregate,
)

# ===================== 组A主对比实验配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 20
LOCAL_EPOCHS = 3
GLOBAL_ROUNDS = 100
LR = 0.01
LR_DECAY = 0.98
EVAL_INTERVAL = 5

# alpha=0.1 时建议每轮更新一次 QGFL 保真度权重
FID_UPDATE_INTERVAL = 1

CALIB_SAMPLES = 500
CALIB_FIXED_SEED = 42
NUM_WORKERS = 8
PIN_MEMORY = True

SEEDS = [42]  # 只跑 1 个 seed
TARGET_ACC = {"cifar10": 85.0, "cifar100": 65.0}

DATASETS = ["cifar10"]
ALPHAS = [0.1]  # 只跑 alpha=0.1
METHODS_MAP = {
    "cifar10": [
     "FedAvg",
     "FedProx",
     "SCAFFOLD",
     "FedNova",
     "FedDyn"
     "QGFL (Ours)"]
}

# ===================== 功能开关配置 =====================
ENABLE_CHECKPOINT = True
CHECKPOINT_INTERVAL = 5
CHECKPOINT_DIR = "results/federated/checkpoints"

ENABLE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 25

ENABLE_ROUND_LOG = True
LOG_DIR = "results/federated/round_logs"

PRINT_INTERVAL = 10

os.makedirs("results/federated", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ===================== 断点保存与加载 =====================
def save_checkpoint(exp_name, round_idx, global_state, global_control_var,
                    acc_history, comm_total, best_acc, best_round, patience_counter,
                    cached_weights=None, cached_avg_fid=0.0):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    torch.save({
        "round_idx": round_idx,
        "global_state": global_state,
        "global_control_var": global_control_var,
        "acc_history": acc_history,
        "comm_total": comm_total,
        "best_acc": best_acc,
        "best_round": best_round,
        "patience_counter": patience_counter,
        "cached_weights": cached_weights,
        "cached_avg_fid": cached_avg_fid
    }, ckpt_path)


def load_checkpoint(exp_name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"🔄 检测到断点，从第 {ckpt['round_idx'] + 1} 轮继续：{exp_name}")
    return ckpt


# ===================== 单组实验运行 =====================
def run_single_experiment(dataset_name, alpha, method, seed):
    exp_name = f"resnet18_{dataset_name}_alpha{alpha}_{method.replace(' ', '_')}_seed{seed}"

    # 1. 数据加载与划分
    train_set, test_set, num_classes = get_datasets(dataset_name)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    client_datasets = dirichlet_non_iid_partition(train_set, NUM_CLIENTS, alpha, seed=seed)
    calib_loader = build_calibration_loader(
        dataset_name,
        num_samples=CALIB_SAMPLES,
        seed=CALIB_FIXED_SEED
    )

    # 2. 初始化客户端
    clients = []
    for i in range(NUM_CLIENTS):
        client = FederatedClient(
            client_id=i,
            dataset=client_datasets[i],
            model_class=ResNet18_CIFAR,
            num_classes=num_classes,
            device=DEVICE,
            lr=LR,
            local_epochs=LOCAL_EPOCHS
        )
        clients.append(client)

    # 3. 初始化全局模型与状态
    global_model = ResNet18_CIFAR(num_classes=num_classes).to(DEVICE)
    global_state = global_model.state_dict()
    global_control_var = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}

    initial_global_state = {k: v.clone() for k, v in global_state.items()}

    start_round = 0
    acc_history = []
    comm_total = 0.0
    best_acc = 0.0
    best_round = 0
    patience_counter = 0
    avg_fidelity = 0.0
    early_stop_triggered = False
    last_test_acc = 0.0

    # QGFL 保真度权重缓存
    cached_weights = None
    cached_avg_fid = 0.0

    # 4. 加载断点
    if ENABLE_CHECKPOINT:
        ckpt = load_checkpoint(exp_name)
        if ckpt is not None:
            start_round = ckpt["round_idx"] + 1
            global_state = ckpt["global_state"]
            global_control_var = ckpt["global_control_var"]
            acc_history = ckpt["acc_history"]
            comm_total = ckpt["comm_total"]
            best_acc = ckpt["best_acc"]
            best_round = ckpt["best_round"]
            patience_counter = ckpt["patience_counter"]
            last_test_acc = acc_history[-1] if acc_history else 0.0
            cached_weights = ckpt.get("cached_weights", None)
            cached_avg_fid = ckpt.get("cached_avg_fid", 0.0)
            global_model.load_state_dict(global_state, strict=False)

    client_weights = np.array([c.num_samples for c in clients], dtype=np.float64)
    client_weights = client_weights / client_weights.sum()

    # 5. 联邦训练主循环
    desc_str = f"{method[:6]:<6} α={alpha} s={seed}"
    pbar = tqdm(
        range(start_round, GLOBAL_ROUNDS),
        desc=desc_str,
        initial=start_round,
        total=GLOBAL_ROUNDS,
        mininterval=1.0
    )

    for r in pbar:
        # --- 本地训练 ---
        client_states = []
        client_deltas = []
        client_control_vars = []

        local_train_method = "fedavg" if method == "QGFL (Ours)" else method.lower()

        for client in clients:
            client.load_global_params(global_state)
            new_state, delta, num_samples, new_ctrl = client.local_train(
                method=local_train_method,
                round_idx=r,
                global_state_dict=global_state,
                global_control_var=global_control_var,
                prox_mu=0.01,
                lr_decay=LR_DECAY
            )
            client_states.append(new_state)
            client_deltas.append(delta)
            client_control_vars.append(new_ctrl)

        # --- 聚合逻辑 ---
        if method == "FedAvg":
            global_state = fedavg_aggregate(client_states, client_weights)
        elif method == "QGFL (Ours)":
            global_state, avg_fidelity, cached_weights = qgfl_aggregate(
                global_state,
                client_states,
                client_weights,
                ResNet18_CIFAR,
                num_classes,
                calib_loader,
                DEVICE,
                gamma=1.15,
                update_fidelity=True,
                cached_weights=cached_weights,
                weight_ema=0.85
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        # 调试：首轮聚合后打印全局参数变化量
        if r == 0:
            total_diff = 0.0
            for name in global_state:
                if global_state[name].dtype == torch.float32:
                    total_diff += torch.norm(global_state[name] - initial_global_state[name]).item()
            print(f"\n�� [调试] 首轮聚合后全局参数变化总量：{total_diff:.6f}")
            print(f"   正常范围：1~10；接近0则聚合完全失效\n")

        # --- 通信量累计 ---
        comm_total += compute_communication_cost(global_state)

        # --- 评估逻辑 ---
        is_eval_round = (r + 1) % EVAL_INTERVAL == 0 or r == 0 or r == GLOBAL_ROUNDS - 1

        if is_eval_round:
            global_model.load_state_dict(global_state, strict=False)
            test_acc = evaluate_model(global_model, test_loader, DEVICE)
            last_test_acc = test_acc

            # --- 早停逻辑 ---
            if ENABLE_EARLY_STOP:
                if test_acc > best_acc:
                    best_acc = test_acc
                    best_round = r + 1
                    patience_counter = 0
                    if ENABLE_CHECKPOINT:
                        save_checkpoint(
                            exp_name + "_best",
                            r,
                            global_state,
                            global_control_var,
                            acc_history,
                            comm_total,
                            best_acc,
                            best_round,
                            patience_counter,
                            cached_weights,
                            avg_fidelity
                        )
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        early_stop_triggered = True
        else:
            test_acc = last_test_acc

        acc_history.append(test_acc)

        # --- 进度条显示 ---
        postfix_dict = {
            "Acc": f"{test_acc:.1f}%",
            "Best": f"{best_acc:.1f}%",
            "Pat": f"{EARLY_STOP_PATIENCE - patience_counter}",
            "Comm": f"{comm_total:.0f}M"
        }

        if early_stop_triggered:
            postfix_dict["状态"] = "已早停"
            pbar.set_postfix(postfix_dict, refresh=True)
            pbar.close()
            break
        else:
            pbar.set_postfix(postfix_dict, refresh=True)

        # --- 周期打印 ---
        if (r + 1) % PRINT_INTERVAL == 0:
            print(f"\n[轮次 {r + 1}] {method} | {dataset_name} | α={alpha} | seed={seed}")
            print(f"  当前准确率: {test_acc:.2f}% | 历史最佳: {best_acc:.2f}% @ 第{best_round}轮")
            print(f"  早停剩余耐心: {EARLY_STOP_PATIENCE - patience_counter} | 累计通信: {comm_total:.2f}MB\n")

        # --- 定时保存断点 ---
        if ENABLE_CHECKPOINT and (r + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                exp_name,
                r,
                global_state,
                global_control_var,
                acc_history,
                comm_total,
                best_acc,
                best_round,
                patience_counter,
                cached_weights,
                avg_fidelity
            )

        # --- 保存每轮日志 ---
        if ENABLE_ROUND_LOG:
            log_row = {
                "round": r + 1,
                "top1_acc": test_acc,
                "comm_MB": comm_total,
                "best_acc": best_acc,
                "is_eval": is_eval_round
            }
            log_path = os.path.join(LOG_DIR, f"{exp_name}_log.csv")
            pd.DataFrame([log_row]).to_csv(
                log_path,
                mode="a",
                header=not os.path.exists(log_path),
                index=False
            )

    # 实验结束，清理普通断点
    if ENABLE_CHECKPOINT:
        normal_ckpt = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
        if os.path.exists(normal_ckpt):
            os.remove(normal_ckpt)

    # 计算达到目标精度的轮次
    rounds_to_target = GLOBAL_ROUNDS
    for idx, acc in enumerate(acc_history):
        if acc >= TARGET_ACC[dataset_name]:
            rounds_to_target = idx + 1
            break

    num_params = sum(p.numel() for p in global_model.parameters()) / 1e6

    return {
        "top1_acc": best_acc,
        "final_acc": acc_history[-1],
        "best_round": best_round,
        "rounds_to_target": rounds_to_target,
        "comm_total_MB": comm_total,
        "params_M": num_params,
        "avg_fidelity": avg_fidelity if method == "QGFL (Ours)" else None,
        "acc_history": acc_history,
        "early_stopped": early_stop_triggered
    }


# ===================== 批量运行所有实验 =====================
if __name__ == "__main__":
    all_results = []

    total_exp = sum(len(METHODS_MAP[d]) for d in DATASETS) * len(ALPHAS) * len(SEEDS)
    print(f"🚀 开始调试运行，共 {total_exp} 组子实验\n")

    for dataset in DATASETS:
        methods = METHODS_MAP[dataset]
        for alpha in ALPHAS:
            for method in methods:
                accs = []
                rounds_list = []
                comm_list = []
                best_rounds = []
                fids = []

                for seed in SEEDS:
                    res = run_single_experiment(dataset, alpha, method, seed)
                    accs.append(res["top1_acc"])
                    rounds_list.append(res["rounds_to_target"])
                    comm_list.append(res["comm_total_MB"])
                    best_rounds.append(res["best_round"])
                    if res["avg_fidelity"] is not None:
                        fids.append(res["avg_fidelity"])

                acc_mean = np.mean(accs)
                acc_std = np.std(accs)
                rounds_mean = np.mean(rounds_list)
                comm_mean = np.mean(comm_list)
                best_round_mean = np.mean(best_rounds)
                fid_mean = np.mean(fids) if fids else None

                result_row = {
                    "Dataset": dataset.upper(),
                    "Method": method,
                    "Alpha": alpha,
                    "Top-1 Acc (%)": f"{acc_mean:.2f}±{acc_std:.2f}",
                    "Best Round (avg)": f"{best_round_mean:.1f}",
                    "Rounds to Target": f"{rounds_mean:.1f}",
                    "Comm Cost (MB)": f"{comm_mean:.2f}",
                    "Params (M)": f"{res['params_M']:.2f}",
                    "Avg Fidelity": f"{fid_mean:.4f}" if fid_mean else "-"
                }
                all_results.append(result_row)
                print(f"\n✅ 完成: {dataset:>8} | α={alpha} | {method:<12} | Acc={acc_mean:.2f}±{acc_std:.2f}%\n")
                print("-" * 80)

    df = pd.DataFrame(all_results)
    df.to_csv("results/federated/调试结果.csv", index=False, encoding="utf-8-sig")
    print("\n🎉 调试运行完成")
    print("📊 结果：results/federated/调试结果.csv")