import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from federated_utils import (
    get_datasets, dirichlet_non_iid_partition, build_calibration_loader,
    ResNet18_CIFAR, FederatedClient, evaluate_model, compute_communication_cost,
)
from aggregators import fedavg_aggregate, qgfl_aggregate
from quantum_metric import get_choi_state, quantum_fidelity

# ===================== 组E 消融实验 ResNet18对齐A组版 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 20
LOCAL_EPOCHS = 3
GLOBAL_ROUNDS = 100
LR = 0.01
LR_DECAY = 0.98
EVAL_INTERVAL = 5
FID_UPDATE_INTERVAL = 1
SEEDS = [42]
DATASET = "cifar10"
ALPHA = 0.1
TARGET_ACC = 85.0

CALIB_SAMPLES = 500
CALIB_FIXED_SEED = 42
NUM_WORKERS = 8
PIN_MEMORY = True

# ① 已由A组跑完，直接复用
RUN_FULL_QGFL = False

ABLATION_VARIANTS = [
    {
        "name": "① QGFL (full)",
        "type": "full_qgfl",
        "desc": "全量QGFL（基准，复用A组ResNet18 seed42）",
        "skip": True
    },
    {
        "name": "② w/o Quantum Alignment",
        "type": "no_quantum_align",
        "desc": "去掉量子态对齐，通道级参数相似度加权",
        "skip": False
    },
    {
        "name": "③ w/o Adaptive Weight",
        "type": "equal_weight",
        "desc": "去掉保真度自适应权重，量子态等权重聚合",
        "skip": False
    }
]

# 新目录，不污染原VGG16的 ablation 结果
RESULT_DIR = "resultsE/federated/ablation_resnet18"
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULT_DIR, "round_logs")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ===================== 消融变体专属聚合函数 =====================
def aggregate_no_quantum_align(client_states, client_weights, model_class, num_classes, calib_loader, device):
    """
    变体②：去掉量子态对齐，仅用通道级参数余弦相似度加权聚合
    """
    target_layers = [name for name, param in client_states[0].items()
                     if ("weight" in name and "bn" not in name and len(param.shape) >= 2)]

    new_global = {}
    num_clients = len(client_states)

    non_target_layers = [name for name in client_states[0] if name not in target_layers]
    for name in non_target_layers:
        param_dtype = client_states[0][name].dtype
        if param_dtype in (torch.float32, torch.float16, torch.float64):
            avg_param = torch.zeros_like(client_states[0][name])
            for i in range(num_clients):
                avg_param += client_states[i][name] * client_weights[i]
            new_global[name] = avg_param
        else:
            new_global[name] = client_states[0][name].clone()

    for name in target_layers:
        ref_param = client_states[0][name].flatten(1)
        ref_norm = torch.norm(ref_param, dim=1, keepdim=True)

        weights = np.ones(num_clients)
        for i in range(1, num_clients):
            curr_param = client_states[i][name].flatten(1)
            curr_norm = torch.norm(curr_param, dim=1, keepdim=True)
            cos_sim = torch.mean(torch.sum(ref_param * curr_param, dim=1) / (ref_norm * curr_norm + 1e-8)).item()
            weights[i] = max(cos_sim, 0.1)

        weights = weights / weights.sum()
        weighted_sum = torch.zeros_like(client_states[0][name])
        for i in range(num_clients):
            weighted_sum += client_states[i][name] * weights[i]
        new_global[name] = weighted_sum

    torch.cuda.empty_cache()
    return new_global


def aggregate_equal_weight(client_states, model_class, num_classes, calib_loader, device,
                           update_fidelity=True, cached_avg_fid=0.0):
    """
    变体③：等权重聚合（等价FedAvg），保真度仅记录不参与加权
    """
    num_clients = len(client_states)
    new_global = {}

    for name in client_states[0]:
        param_dtype = client_states[0][name].dtype
        if param_dtype in (torch.float32, torch.float16, torch.float64):
            avg_param = torch.zeros_like(client_states[0][name])
            for i in range(num_clients):
                avg_param += client_states[i][name]
            avg_param /= num_clients
            new_global[name] = avg_param
        else:
            new_global[name] = client_states[0][name].clone()

    if not update_fidelity:
        return new_global, cached_avg_fid

    temp_model = model_class(num_classes=num_classes).to(device)
    target_layers = [name.rsplit(".", 1)[0] for name, param in client_states[0].items()
                     if ("weight" in name and "bn" not in name and len(param.shape) >= 2)]
    all_fids = []

    temp_model.load_state_dict(new_global, strict=False)
    global_psi_dict = {}
    for layer_name in target_layers:
        global_psi_dict[layer_name] = get_choi_state(temp_model, layer_name, calib_loader, device)

    for i in range(num_clients):
        temp_model.load_state_dict(client_states[i], strict=False)
        for layer_name in target_layers:
            client_psi = get_choi_state(temp_model, layer_name, calib_loader, device)
            fid = quantum_fidelity(global_psi_dict[layer_name], client_psi).item()
            all_fids.append(fid)
            del client_psi
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    avg_fid = np.mean(all_fids) if all_fids else 0.0
    del temp_model, global_psi_dict
    torch.cuda.empty_cache()
    return new_global, avg_fid


# ===================== 断点保存与加载 =====================
def save_checkpoint(exp_name, round_idx, global_state, acc_history, comm_total, best_acc, best_round, patience_counter):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    torch.save({
        "round_idx": round_idx,
        "global_state": global_state,
        "acc_history": acc_history,
        "comm_total": comm_total,
        "best_acc": best_acc,
        "best_round": best_round,
        "patience_counter": patience_counter
    }, ckpt_path)


def load_checkpoint(exp_name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    print(f"检测到断点，从第 {ckpt['round_idx'] + 1} 轮继续：{exp_name}")
    return ckpt


# ===================== 单组消融实验运行 =====================
def run_ablation_single(variant_type, variant_name, seed):
    exp_name = f"{variant_type}_seed{seed}"

    train_set, test_set, num_classes = get_datasets(DATASET)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    client_datasets = dirichlet_non_iid_partition(train_set, NUM_CLIENTS, ALPHA, seed=seed)
    calib_loader = build_calibration_loader(
        DATASET, num_samples=CALIB_SAMPLES, seed=CALIB_FIXED_SEED
    )

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

    global_model = ResNet18_CIFAR(num_classes=num_classes).to(DEVICE)
    global_state = global_model.state_dict()

    start_round = 0
    acc_history = []
    comm_total = 0.0
    best_acc = 0.0
    best_round = 0
    patience_counter = 0
    early_stop_triggered = False
    last_test_acc = 0.0
    avg_fidelity = 0.0
    cached_weights = None

    ckpt = load_checkpoint(exp_name)
    if ckpt is not None:
        start_round = ckpt["round_idx"] + 1
        global_state = ckpt["global_state"]
        acc_history = ckpt["acc_history"]
        comm_total = ckpt["comm_total"]
        best_acc = ckpt["best_acc"]
        best_round = ckpt["best_round"]
        patience_counter = ckpt["patience_counter"]
        last_test_acc = acc_history[-1] if acc_history else 0.0
        global_model.load_state_dict(global_state, strict=False)

    client_weights = np.array([c.num_samples for c in clients], dtype=np.float64)
    client_weights = client_weights / client_weights.sum()

    desc_str = f"{variant_name[:12]:<12} s={seed}"
    pbar = tqdm(
        range(start_round, GLOBAL_ROUNDS),
        desc=desc_str,
        initial=start_round,
        total=GLOBAL_ROUNDS,
        mininterval=1.0
    )

    for r in pbar:
        client_states = []
        for client in clients:
            client.load_global_params(global_state)
            ret = client.local_train(method="fedavg", round_idx=r, lr_decay=LR_DECAY)
            new_state = ret[0]
            client_states.append(new_state)

        if variant_type == "full_qgfl":
            update_fid = (r + 1) % FID_UPDATE_INTERVAL == 0 or r == 0
            global_state, avg_fidelity, cached_weights = qgfl_aggregate(
                global_state, client_states, client_weights,
                ResNet18_CIFAR, num_classes, calib_loader, DEVICE,
                gamma=1.15, update_fidelity=update_fid,
                cached_weights=cached_weights, weight_ema=0.85
            )
        elif variant_type == "no_quantum_align":
            global_state = aggregate_no_quantum_align(
                client_states, client_weights, ResNet18_CIFAR, num_classes, calib_loader, DEVICE
            )
        elif variant_type == "equal_weight":
            update_fid = (r + 1) % FID_UPDATE_INTERVAL == 0 or r == 0
            global_state, avg_fidelity = aggregate_equal_weight(
                client_states, ResNet18_CIFAR, num_classes, calib_loader, DEVICE,
                update_fidelity=update_fid, cached_avg_fid=avg_fidelity
            )

        comm_total += compute_communication_cost(global_state)

        is_eval_round = (r + 1) % EVAL_INTERVAL == 0 or r == 0 or r == GLOBAL_ROUNDS - 1
        if is_eval_round:
            global_model.load_state_dict(global_state, strict=False)
            test_acc = evaluate_model(global_model, test_loader, DEVICE)
            last_test_acc = test_acc

            if test_acc > best_acc:
                best_acc = test_acc
                best_round = r + 1
                patience_counter = 0
                save_checkpoint(
                    exp_name + "_best",
                    r,
                    global_state,
                    acc_history,
                    comm_total,
                    best_acc,
                    best_round,
                    patience_counter
                )
            else:
                patience_counter += 1
                if patience_counter >= 25:
                    early_stop_triggered = True
        else:
            test_acc = last_test_acc

        acc_history.append(test_acc)

        postfix = {"Acc": f"{test_acc:.1f}%", "Best": f"{best_acc:.1f}%", "Comm": f"{comm_total:.0f}M"}
        if early_stop_triggered:
            postfix["状态"] = "已早停"
            pbar.set_postfix(postfix, refresh=True)
            pbar.close()
            break
        else:
            pbar.set_postfix(postfix, refresh=True)

        if (r + 1) % 10 == 0:
            print(f"\n[轮次 {r + 1}] {variant_name} | seed={seed}")
            print(f"  当前准确率: {test_acc:.2f}% | 历史最佳: {best_acc:.2f}% @ 第{best_round}轮\n")

        if (r + 1) % 5 == 0:
            save_checkpoint(exp_name, r, global_state, acc_history, comm_total, best_acc, best_round, patience_counter)

        log_row = {"round": r + 1, "top1_acc": test_acc, "best_acc": best_acc, "comm_MB": comm_total}
        log_path = os.path.join(LOG_DIR, f"{exp_name}_log.csv")
        pd.DataFrame([log_row]).to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)

    normal_ckpt = os.path.join(CHECKPOINT_DIR, f"{exp_name}.pt")
    if os.path.exists(normal_ckpt):
        os.remove(normal_ckpt)

    rounds_to_target = GLOBAL_ROUNDS
    for idx, acc in enumerate(acc_history):
        if acc >= TARGET_ACC:
            rounds_to_target = idx + 1
            break

    return {
        "variant": variant_name,
        "top1_acc_best": best_acc,
        "best_round": best_round,
        "rounds_to_target": rounds_to_target,
        "comm_total_MB": comm_total,
        "avg_fidelity": avg_fidelity if variant_type != "no_quantum_align" else "-",
        "acc_history": acc_history
    }


# ===================== 主程序入口 =====================
if __name__ == "__main__":
    all_results = []
    run_variants = [v for v in ABLATION_VARIANTS if not v["skip"]]
    total_exp = len(run_variants) * len(SEEDS)
    print(f"开始组E消融实验(ResNet18对齐A组)，实际运行 {total_exp} 组（①复用A组）\n")

    for variant in ABLATION_VARIANTS:
        if variant["skip"]:
            placeholder_row = {
                "变体": variant["name"],
                "验证目的": variant["desc"],
                "Top-1准确率(%)": "84.09 (复用 A/汇总表.csv QGFL alpha0.1 seed42)",
                "最佳轮次(平均)": "85.0",
                "通信量(MB)": "4266.21",
                "平均保真度": "-"
            }
            all_results.append(placeholder_row)
            print(f"跳过: {variant['name']}（已复用A组 84.09）\n")
            continue

        accs = []
        best_rounds = []
        comms = []
        fids = []

        for seed in SEEDS:
            res = run_ablation_single(variant["type"], variant["name"], seed)
            accs.append(res["top1_acc_best"])
            best_rounds.append(res["best_round"])
            comms.append(res["comm_total_MB"])
            if res["avg_fidelity"] != "-":
                fids.append(res["avg_fidelity"])

        acc_mean = np.mean(accs)
        acc_std = np.std(accs)
        br_mean = np.mean(best_rounds)
        comm_mean = np.mean(comms)
        fid_mean = np.mean(fids) if fids else "-"

        row = {
            "变体": variant["name"],
            "验证目的": variant["desc"],
            "Top-1准确率(%)": f"{acc_mean:.2f}±{acc_std:.2f}",
            "最佳轮次(平均)": f"{br_mean:.1f}",
            "通信量(MB)": f"{comm_mean:.2f}",
            "平均保真度": f"{fid_mean:.4f}" if fid_mean != "-" else "-"
        }
        all_results.append(row)
        print(f"\n完成: {variant['name']} | 准确率={acc_mean:.2f}±{acc_std:.2f}%\n")
        print("-" * 80)

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(RESULT_DIR, "消融实验对比表.csv"), index=False, encoding="utf-8-sig")
    df.to_excel(os.path.join(RESULT_DIR, "消融实验对比表.xlsx"), index=False)

    print("\n消融实验运行完成！")
    print(f"结果表：{RESULT_DIR}/消融实验对比表.xlsx（①已复用A组，无需重跑）")