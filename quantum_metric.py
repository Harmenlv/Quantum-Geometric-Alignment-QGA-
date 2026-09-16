import torch
import torch.nn as nn

def get_choi_state(model, layer_name, calibration_loader, device):
    """
    钩子式中间层响应提取 + 2x2 GAP降维
    优化版：保留少量空间维度信息，算子特征更丰富，保真度计算更准确
    24G显存环境下无压力，精度提升明显
    """
    model.eval()
    module_dict = dict(model.named_modules())
    target_module = module_dict[layer_name]
    is_conv = isinstance(target_module, nn.Conv2d)

    cov_sum = None
    total_samples = 0

    def forward_hook(module, input, output):
        nonlocal cov_sum, total_samples
        with torch.no_grad():
            if is_conv:
                # 优化：2x2平均池化，保留空间信息，特征维度从C变为4*C
                feat = nn.functional.adaptive_avg_pool2d(output, 2).flatten(start_dim=1)
            else:
                feat = output.flatten(start_dim=1)

            batch_size = feat.shape[0]
            total_samples += batch_size

            mean_batch = feat.mean(dim=0, keepdim=True)
            centered_batch = feat - mean_batch
            batch_cov = centered_batch.T @ centered_batch

            if cov_sum is None:
                cov_sum = batch_cov
            else:
                cov_sum += batch_cov

    hook_handle = target_module.register_forward_hook(forward_hook)

    with torch.no_grad():
        for inputs, _ in calibration_loader:
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs)

    hook_handle.remove()

    # 数值稳定 + 密度矩阵 + 特征分解
    cov_matrix = cov_sum / total_samples
    dim = cov_matrix.shape[0]
    cov_matrix += 1e-8 * torch.eye(dim, device=device)

    rho = cov_matrix / torch.trace(cov_matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(rho)
    psi = eigenvectors[:, -1]
    psi = psi / torch.norm(psi)

    del cov_sum, cov_matrix, rho, eigenvalues, eigenvectors
    torch.cuda.empty_cache()

    return psi

def quantum_fidelity(psi1, psi2):
    """纯态量子保真度 F = |⟨ψ1|ψ2⟩|²"""
    return torch.abs(torch.dot(psi1.conj(), psi2)) ** 2

def fubini_study_distance(psi1, psi2):
    """Fubini-Study 距离 d_FS = arccos(√F)"""
    fid = quantum_fidelity(psi1, psi2).clamp(0.0, 1.0)
    return torch.acos(torch.sqrt(fid))