import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== 输出级量子保真度度量 =====================
# 与 paper quantum_metric.py 中 get_choi_state 一致的计算方式，
# 应用于输出层（logits）而非中间层。

@torch.no_grad()
def _get_output_choi_state(model, calib_loader, device):
    """
    计算模型输出层（logits）的 Choi 态（量子纯态 ψ）：
    1. 收集所有校准集样本上的输出 logits → [N, K]
    2. 计算协方差矩阵（K×K）
    3. 取最大特征值的特征向量，归一化 → 量子纯态 ψ

    返回: psi (Tensor, CPU)
    """
    model.eval()
    all_outputs = []

    for inputs, _ in calib_loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        all_outputs.append(outputs.detach().cpu())

    outputs = torch.cat(all_outputs, dim=0)          # [N, K]
    N, K = outputs.shape

    mean = outputs.mean(dim=0, keepdim=True)
    centered = outputs - mean
    cov = (centered.T @ centered) / (N - 1 + 1e-8)
    cov += 1e-8 * torch.eye(K, device=cov.device)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    psi = eigenvectors[:, -1]                        # 最大特征值对应特征向量
    psi = psi / (torch.norm(psi) + 1e-8)

    return psi


@torch.no_grad()
def _get_ensemble_output_state(calib_loader, device, client_models, weights=None):
    """
    计算指定权重集成模型的输出 Choi 态。
    权重缺省为均匀；实际调用时传数据量权重（= FedDF 的 teacher）。
    """
    num_clients = len(client_models)
    if weights is None:
        weights = np.ones(num_clients, dtype=np.float64) / num_clients
    else:
        weights = np.asarray(weights, dtype=np.float64)
        weights = weights / weights.sum()

    all_ensemble_outputs = []

    for inputs, _ in calib_loader:
        inputs = inputs.to(device, non_blocking=True)

        ensemble_logits = None
        for i, model in enumerate(client_models):
            model.eval()
            logits = model(inputs)
            if ensemble_logits is None:
                ensemble_logits = weights[i] * logits
            else:
                ensemble_logits += weights[i] * logits

        all_ensemble_outputs.append(ensemble_logits.detach().cpu())

    outputs = torch.cat(all_ensemble_outputs, dim=0)  # [N, K]
    N, K = outputs.shape

    mean = outputs.mean(dim=0, keepdim=True)
    centered = outputs - mean
    cov = (centered.T @ centered) / (N - 1 + 1e-8)
    cov += 1e-8 * torch.eye(K, device=cov.device)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    psi = eigenvectors[:, -1]
    psi = psi / (torch.norm(psi) + 1e-8)

    return psi


@torch.no_grad()
def _quantum_fidelity(psi1, psi2):
    """纯态量子保真度 F = |⟨ψ1|ψ2⟩|²"""
    return torch.abs(torch.dot(psi1.conj(), psi2)) ** 2


# ===================== 门控辅助 =====================

@torch.no_grad()
def _ensemble_acc_on_loader(client_models, loader, device, weights):
    """计算给定权重下客户端集成在 loader 上的准确率（用于门控比较）"""
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        agg_logits = None
        for i, model in enumerate(client_models):
            model.eval()
            logits = model(inputs)
            if agg_logits is None:
                agg_logits = weights[i] * logits
            else:
                agg_logits += weights[i] * logits

        pred = agg_logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / max(total, 1)


# ===================== 核心：输出级量子保真度权重估计（修正版） =====================

@torch.no_grad()
def compute_quantum_fidelity_weights(
    client_models,
    calib_loader,
    device,
    reference_weights=None,
    round_idx=0,
    warmup_rounds=10,
    old_weights=None,
    ema_momentum=0.95,
    uniform_blend=0.30,
    qgfl_bias=0.35,
    qgfl_scale=0.65,
    gate=True,
):
    """
    基于输出级量子保真度的客户端权重估计（修正版）。

    关键修正：参考集成从"均匀"改为"数据量权重"（= FedDF 的 teacher），
    使保真度信号是相对 FedDF 的增量，而非相对均匀的增量。

    流程：
    1. warmup 轮次直接返回数据量权重（= FedDF）
    2. 每个客户端算输出 Choi 态 ψ_i
    3. 数据量加权集成的输出 Choi 态 ψ_ref
    4. 保真度 F_i = |⟨ψ_i|ψ_ref⟩|²
    5. z-score 标准化后套 QGFL 公式（增强低维信号的区分度）
    6. 与数据量权重混合 + EMA 平滑
    7. 门控：加权 teacher 不优于数据量 teacher 时回退数据量权重（= FedDF）

    返回:
      - weights: list[float]
      - fidelity_list: list[float]
      - weighted_calib_acc: float
      - ref_calib_acc: float
      - fell_back: bool   是否回退到数据量权重
      - is_active: bool   warmup 后为 True
    """
    num_clients = len(client_models)
    if reference_weights is None:
        reference = np.ones(num_clients, dtype=np.float64) / num_clients
    else:
        reference = np.asarray(reference_weights, dtype=np.float64)
        reference = reference / reference.sum()

    # warmup：直接就是 FedDF
    if round_idx < warmup_rounds:
        return reference.tolist(), [], 0.0, 0.0, False, False

    # 1) 每个客户端的输出 Choi 态
    client_states = []
    for model in client_models:
        psi = _get_output_choi_state(model, calib_loader, device)
        client_states.append(psi)

    # 2) 数据量加权集成的输出 Choi 态（参考态 = FedDF teacher 态）
    ensemble_state = _get_ensemble_output_state(
        calib_loader, device, client_models, weights=reference
    )

    # 3) 量子保真度
    fidelity_values = []
    for psi in client_states:
        F = _quantum_fidelity(psi, ensemble_state)
        fidelity_values.append(F.item())

    fidelity_values = np.array(fidelity_values, dtype=np.float64)
    fidelity_values = np.clip(fidelity_values, 1e-8, 1.0)

    # 4) z-score 标准化，再套 QGFL 公式（w = bias + scale·√score）
    std_fid = fidelity_values.std()
    if std_fid > 1e-6:
        z = (fidelity_values - fidelity_values.mean()) / std_fid
    else:
        z = np.zeros_like(fidelity_values)
    z = np.clip(z, -2.0, 2.0)

    raw_scores = qgfl_bias + qgfl_scale * np.sqrt(np.clip(z, 0.0, None))
    raw_weights = raw_scores / raw_scores.sum()

    # 5) 与数据量权重混合（而非均匀）
    blended = (1.0 - uniform_blend) * raw_weights + uniform_blend * reference

    # 6) EMA 平滑
    if old_weights is not None:
        old_weights = np.asarray(old_weights, dtype=np.float64)
        if old_weights.shape == blended.shape:
            blended = ema_momentum * old_weights + (1.0 - ema_momentum) * blended

    blended = blended / blended.sum()

    # 7) 门控：比数据量 teacher 差就回退（= FedDF）
    weighted_calib_acc = _ensemble_acc_on_loader(client_models, calib_loader, device, blended)
    ref_calib_acc = _ensemble_acc_on_loader(client_models, calib_loader, device, reference)

    fell_back = False
    if gate and weighted_calib_acc < ref_calib_acc:
        blended = reference.copy()
        fell_back = True

    return (
        blended.tolist(),
        fidelity_values.tolist(),
        weighted_calib_acc,
        ref_calib_acc,
        fell_back,
        True,
    )


# ===================== 服务器蒸馏 =====================

def distill_server_model(
    server_model,
    client_models,
    calib_loader,
    device,
    client_weights=None,
    temperature=5.0,
    distill_epochs=20,
    lr=0.01,
    hard_weight=0.1,
    ema_decay=0.95,
):
    """
    异构场景下的服务器蒸馏（FedDF 强化版）。
    teacher = client_weights 加权平均（数据量权重或量子保真度权重）。
    """
    server_model = server_model.to(device)
    old_state = copy.deepcopy(server_model.state_dict())

    num_clients = len(client_models)
    if client_weights is None:
        client_weights = np.ones(num_clients, dtype=np.float64) / num_clients
    else:
        client_weights = np.asarray(client_weights, dtype=np.float64)
        client_weights = client_weights / client_weights.sum()

    optimizer = torch.optim.SGD(
        server_model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()

    server_model.train()
    for _ in range(distill_epochs):
        for inputs, labels in calib_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = None
                for i, model in enumerate(client_models):
                    model.eval()
                    logits = model(inputs)
                    if teacher_logits is None:
                        teacher_logits = client_weights[i] * logits
                    else:
                        teacher_logits += client_weights[i] * logits

            optimizer.zero_grad()
            student_logits = server_model(inputs)

            soft_loss = kl_loss(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
            ) * (temperature * temperature)

            hard_loss = ce_loss(student_logits, labels)
            loss = (1.0 - hard_weight) * soft_loss + hard_weight * hard_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=5.0)
            optimizer.step()

    # EMA 合并新旧知识
    new_state = server_model.state_dict()
    merged_state = {}
    for k, v in new_state.items():
        if torch.is_tensor(v) and torch.is_floating_point(v) and k in old_state and old_state[k].shape == v.shape:
            merged_state[k] = ema_decay * v + (1.0 - ema_decay) * old_state[k]
        else:
            merged_state[k] = v

    server_model.load_state_dict(merged_state, strict=False)
    server_model.eval()
    torch.cuda.empty_cache()
    return server_model


# ===================== 集成评估 =====================

@torch.no_grad()
def evaluate_ensemble(client_models, test_loader, device, client_weights=None):
    """客户端模型集成评估（用于 ensemble 基线）"""
    num_clients = len(client_models)
    if client_weights is None:
        client_weights = np.ones(num_clients, dtype=np.float64) / num_clients
    else:
        client_weights = np.asarray(client_weights, dtype=np.float64)
        client_weights = client_weights / client_weights.sum()

    correct = 0
    total = 0

    for inputs, labels in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        agg_logits = None
        for i, model in enumerate(client_models):
            model.eval()
            logits = model(inputs)
            if agg_logits is None:
                agg_logits = client_weights[i] * logits
            else:
                agg_logits += client_weights[i] * logits

        pred = agg_logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total