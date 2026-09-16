import torch
import torch.nn as nn
import numpy as np
from quantum_metric import get_choi_state, quantum_fidelity

def _is_float_tensor(x):
    return torch.is_tensor(x) and torch.is_floating_point(x)

def _normalize_weights(weights, fallback_len):
    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        return np.ones(fallback_len, dtype=np.float64) / fallback_len
    arr = np.clip(arr, 1e-8, None)
    s = arr.sum()
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(arr, dtype=np.float64) / len(arr)
    return arr / s

# ===================== 基础基线聚合函数 =====================
def fedavg_aggregate(client_states, client_weights):
    new_global = {}
    num_clients = len(client_states)
    for name in client_states[0]:
        param_dtype = client_states[0][name].dtype
        if param_dtype in (torch.float32, torch.float16, torch.float64):
            weighted_sum = torch.zeros_like(client_states[0][name])
            for i in range(num_clients):
                weighted_sum += client_states[i][name] * client_weights[i]
            new_global[name] = weighted_sum
        else:
            new_global[name] = client_states[0][name].clone()
    return new_global

def fednova_aggregate(global_state, client_states, client_deltas, client_weights, local_epochs, lr):
    """
    稳定版 FedNova：
    - 参数项：直接按 delta 聚合，不再除以 lr
    - BN running_mean / running_var：做加权平均
    - 整数 buffer：直接保留
    """
    new_global = {}
    num_clients = len(client_states)
    delta_keys = set(client_deltas[0].keys())

    for name in global_state:
        tensor = global_state[name]

        if torch.is_tensor(tensor) and torch.is_floating_point(tensor):
            if name in delta_keys:
                weighted_delta = torch.zeros_like(tensor)
                for i in range(num_clients):
                    weighted_delta += client_deltas[i][name] * client_weights[i]
                new_global[name] = tensor + weighted_delta
            else:
                weighted_avg = torch.zeros_like(tensor)
                for i in range(num_clients):
                    weighted_avg += client_states[i][name] * client_weights[i]
                new_global[name] = weighted_avg
        else:
            new_global[name] = tensor.clone() if torch.is_tensor(tensor) else tensor

    return new_global

def scaffold_aggregate(global_state, global_control_var, client_deltas, client_control_vars, client_weights):
    new_global = {}
    new_control_var = {}
    num_clients = len(client_deltas)
    delta_keys = set(client_deltas[0].keys()) if num_clients > 0 else set()

    for name in global_state:
        param_dtype = global_state[name].dtype
        is_float = param_dtype in (torch.float32, torch.float16, torch.float64)

        if is_float and name in delta_keys:
            # --- 参数项：SCAFFOLD delta 聚合 ---
            weighted_delta = torch.zeros_like(global_state[name])
            for i in range(num_clients):
                weighted_delta += client_deltas[i][name] * client_weights[i]
            new_global[name] = global_state[name] + weighted_delta

            # 控制变量同样加权平均
            weighted_ctrl = torch.zeros_like(global_control_var[name])
            for i in range(len(client_control_vars)):
                weighted_ctrl += client_control_vars[i][name] * client_weights[i]
            new_control_var[name] = weighted_ctrl

        elif is_float:
            # --- BN buffer（running_mean / running_var 等）：直接保留全局状态 ---
            new_global[name] = global_state[name].clone()
            if name in global_control_var:
                new_control_var[name] = global_control_var[name].clone()

        else:
            # --- 整数 buffer（num_batches_tracked）：直接保留 ---
            new_global[name] = global_state[name].clone()
            if name in global_control_var:
                new_control_var[name] = global_control_var[name].clone()

    return new_global, new_control_var

def feddyn_aggregate(client_states, client_weights, alpha_dyn=0.01):
    new_global = {}
    num_clients = len(client_states)
    for name in client_states[0]:
        param_dtype = client_states[0][name].dtype
        if param_dtype in (torch.float32, torch.float16, torch.float64):
            avg_param = torch.zeros_like(client_states[0][name])
            for i in range(num_clients):
                avg_param += client_states[i][name] * client_weights[i]
            new_global[name] = avg_param
        else:
            new_global[name] = client_states[0][name].clone()
    return new_global

def feddf_aggregate(
    global_model,
    client_states,
    calibration_loader,
    device,
    num_classes,
    distill_epochs=6,
    temperature=3.0,
    lr=0.003
):
    """
    稳定版 FedDF：
    1) 先在 calibration set 上计算每个 batch 的教师 logits 平均
    2) 再用多轮蒸馏更新 global_model
    3) 使用 temperature 提升蒸馏信号强度
    """
    global_model = global_model.to(device)
    temp_model = type(global_model)(num_classes=num_classes).to(device)

    num_clients = len(client_states)
    teacher_batches = []

    # 1. 计算 teacher logits（按 batch 平均）
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device, non_blocking=True)
            batch_sum_logits = None

            for client_state in client_states:
                temp_model.load_state_dict(client_state, strict=False)
                temp_model.eval()
                logits = temp_model(inputs)

                if batch_sum_logits is None:
                    batch_sum_logits = logits
                else:
                    batch_sum_logits += logits

            teacher_batches.append(batch_sum_logits / num_clients)

    # 2. 蒸馏更新 global_model
    optimizer = torch.optim.SGD(
        global_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    criterion = nn.KLDivLoss(reduction="batchmean")

    global_model.train()
    for _ in range(distill_epochs):
        for batch_idx, (inputs, _) in enumerate(calibration_loader):
            inputs = inputs.to(device, non_blocking=True)
            teacher_logits = teacher_batches[batch_idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            student_logits = global_model(inputs)

            loss = criterion(
                torch.log_softmax(student_logits / temperature, dim=1),
                torch.softmax(teacher_logits / temperature, dim=1)
            ) * (temperature * temperature)

            loss.backward()
            optimizer.step()

    global_model.eval()
    del temp_model, teacher_batches
    torch.cuda.empty_cache()
    return global_model.state_dict()

# ===================== QGFL 核心聚合函数（ResNet18 更稳版本） =====================
def qgfl_aggregate(
    global_state,
    client_states,
    client_weights,
    model_class,
    num_classes,
    calibration_loader,
    device,
    gamma=1.0,
    update_fidelity=True,
    cached_weights=None,
    weight_ema=0.85
):
    """
    更稳的 QGFL 版本，适合 ResNet18 / CIFAR 场景：
    1) 只对卷积/全连接权重做 QGFL
    2) 保真度权重做平滑和归一化
    3) 目标层同时保留 FedAvg 路径，降低非IID下的抖动
    """

    # 只选真正的浮点权重参数，排除 BN buffer / 统计量
    target_layers = [
        name for name, param in global_state.items()
        if _is_float_tensor(param)
        and param.ndim >= 2
        and "weight" in name.lower()
        and "bn" not in name.lower()
    ]
    target_module_names = [name.rsplit(".", 1)[0] for name in target_layers]
    num_clients = len(client_states)

    # ===================== 步骤1：计算或更新保真度权重 =====================
    if update_fidelity or cached_weights is None:
        temp_model = model_class(num_classes=num_classes).to(device)
        temp_model.load_state_dict(global_state, strict=False)

        global_psi_dict = {}
        for layer_name in target_module_names:
            global_psi_dict[layer_name] = get_choi_state(
                temp_model, layer_name, calibration_loader, device
            )

        new_weights_dict = {name: [] for name in target_layers}
        all_fids = []

        for client_idx in range(num_clients):
            temp_model.load_state_dict(client_states[client_idx], strict=False)

            for layer_idx, param_name in enumerate(target_layers):
                module_name = target_module_names[layer_idx]
                client_psi = get_choi_state(
                    temp_model, module_name, calibration_loader, device
                )
                fid = quantum_fidelity(global_psi_dict[module_name], client_psi).item()
                fid = float(np.clip(fid, 0.0, 1.0))
                all_fids.append(fid)

                # 更稳的映射：避免极低保真度把权重压得太狠
                weight = 0.35 + 0.65 * np.sqrt(fid)
                new_weights_dict[param_name].append(weight)

                del client_psi

        avg_fidelity = float(np.mean(all_fids)) if all_fids else 0.0
        del temp_model, global_psi_dict
        torch.cuda.empty_cache()

        # 如果有旧缓存，则做指数平滑
        if cached_weights is not None:
            weights_dict = {}
            for name in target_layers:
                new_w = np.asarray(new_weights_dict[name], dtype=np.float64)
                old_w = np.asarray(cached_weights.get(name, []), dtype=np.float64)

                if old_w.shape == new_w.shape and old_w.size > 0:
                    smoothed_w = weight_ema * new_w + (1.0 - weight_ema) * old_w
                else:
                    smoothed_w = new_w

                weights_dict[name] = _normalize_weights(smoothed_w, num_clients).tolist()
        else:
            weights_dict = {}
            for name in target_layers:
                weights_dict[name] = _normalize_weights(new_weights_dict[name], num_clients).tolist()
    else:
        weights_dict = cached_weights
        avg_fidelity = 0.0

    # ===================== 步骤2：分层聚合 =====================
    new_global = {}

    # 非目标层直接 FedAvg
    non_target_layers = [name for name in global_state if name not in target_layers]
    for name in non_target_layers:
        param_dtype = global_state[name].dtype
        if param_dtype in (torch.float32, torch.float16, torch.float64):
            avg_param = torch.zeros_like(global_state[name])
            for i in range(num_clients):
                avg_param += client_states[i][name] * client_weights[i]
            new_global[name] = avg_param
        else:
            new_global[name] = client_states[0][name].clone()

    # 目标层：QGFL + FedAvg 融合，稳定很多
    for name in target_layers:
        w_global = global_state[name]

        # FedAvg 路径
        fedavg_param = torch.zeros_like(w_global)
        for i in range(num_clients):
            fedavg_param += client_states[i][name] * client_weights[i]

        # QGFL 路径
        weighted_delta_sum = torch.zeros_like(w_global)
        weight_sum = 0.0

        for i in range(num_clients):
            delta = client_states[i][name] - w_global
            layer_client_weight = weights_dict[name][i]
            combined_weight = float(client_weights[i] * layer_client_weight)
            weighted_delta_sum += combined_weight * delta
            weight_sum += combined_weight

        if weight_sum > 0:
            avg_delta = weighted_delta_sum / weight_sum
            mean_layer_conf = float(np.mean(weights_dict[name]))

            # 根据层置信度自适应缩放，减少抖动
            adaptive_gamma = gamma * (0.80 + 0.20 * mean_layer_conf)
            qgfl_param = w_global + adaptive_gamma * avg_delta

            # 最终再和 FedAvg 融合一次，稳很多
            # 置信度越高，越偏向 QGFL
            blend = float(np.clip(0.45 + 0.35 * mean_layer_conf, 0.45, 0.80))
            new_global[name] = blend * qgfl_param + (1.0 - blend) * fedavg_param
        else:
            new_global[name] = fedavg_param

    return new_global, avg_fidelity, weights_dict