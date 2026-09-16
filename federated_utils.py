import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np

# ===================== 数据集相关 =====================
def get_datasets(dataset_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_set, test_set, num_classes

def dirichlet_non_iid_partition(dataset, num_clients, alpha, seed=42):
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(set(targets))

    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    class_indices = [np.where(targets == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        proportions = label_distribution[c]
        split_sizes = (proportions * len(class_indices[c])).astype(int)
        split_sizes[-1] = len(class_indices[c]) - split_sizes[:-1].sum()
        np.random.shuffle(class_indices[c])
        start = 0
        for i in range(num_clients):
            end = start + split_sizes[i]
            client_indices[i].extend(class_indices[c][start:end])
            start = end

    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    return client_datasets

def build_calibration_loader(dataset_name, num_samples=500, seed=42, batch_size=128):
    np.random.seed(seed)

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        base_dataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        )
    elif dataset_name == "cifar100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        base_dataset = torchvision.datasets.CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        )
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    total = len(base_dataset)
    num_samples = min(num_samples, total)
    indices = np.random.choice(total, num_samples, replace=False)
    calib_set = Subset(base_dataset, indices)

    loader = DataLoader(
        calib_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return loader

# ===================== ResNet18 适配 CIFAR =====================
class ResNet18_CIFAR(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        # CIFAR 更适合 3x3 / stride=1
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out", nonlinearity="relu")

        # CIFAR 不需要 ImageNet 的 maxpool
        self.maxpool = nn.Identity()

        # 分类头按类别数重建
        self.fc = nn.Linear(512, num_classes)

# ===================== 联邦客户端类 =====================
class FederatedClient:
    def __init__(self, client_id, dataset, model_class, num_classes, device, lr=0.01, local_epochs=2):
        self.client_id = client_id
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

        self.model = model_class(num_classes=num_classes).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.0, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.loader = DataLoader(
            dataset, batch_size=64, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=False
        )

        self.control_var = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

    def load_global_params(self, global_state_dict):
        self.model.load_state_dict(global_state_dict, strict=False)

    def local_train(self, method="fedavg", round_idx=0, global_state_dict=None,
                    global_control_var=None, prox_mu=0.01, lr_decay=0.98):
        self.model.train()

        current_lr = self.lr * (lr_decay ** round_idx)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
            if method == "scaffold":
                param_group["momentum"] = 0.0

        prev_state = {name: param.detach().clone() for name, param in self.model.named_parameters()}

        for epoch in range(self.local_epochs):
            for inputs, labels in self.loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if method == "fedprox" and global_state_dict is not None:
                    prox_loss = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_state_dict:
                            global_param = global_state_dict[name].to(self.device)
                            prox_loss += torch.norm(param - global_param) ** 2
                    loss += (prox_mu / 2) * prox_loss

                if method == "scaffold" and global_control_var is not None:
                    ctrl_loss = 0.0
                    for name, param in self.model.named_parameters():
                        c_local = self.control_var[name]
                        c_global = global_control_var[name]
                        ctrl_loss += torch.sum(param * (c_global - c_local))
                    loss += ctrl_loss

                loss.backward()
                self.optimizer.step()

        new_state = self.model.state_dict()

        delta = {}
        for name in prev_state:
            if prev_state[name].dtype in (torch.float32, torch.float16, torch.float64):
                delta[name] = (new_state[name] - prev_state[name]).detach()
            else:
                delta[name] = torch.zeros_like(prev_state[name])

        new_ctrl = {k: v.detach().clone() for k, v in self.control_var.items()}
        if method == "scaffold" and global_control_var is not None:
            new_ctrl = {}
            for name in self.control_var:
                if prev_state[name].dtype in (torch.float32, torch.float16, torch.float64):
                    new_ctrl[name] = (
                            self.control_var[name]
                            - global_control_var[name]
                            - delta[name] / (self.local_epochs * current_lr)
                    ).detach()
                else:
                    new_ctrl[name] = self.control_var[name].detach().clone()

            self.control_var = {k: v.detach().clone() for k, v in new_ctrl.items()}

        if self.client_id == 0:
            param_change = sum(
                torch.norm(delta[n]).item()
                for n in delta
                if delta[n].dtype == torch.float32
            )
            print(f"\n🔥 [调试] 客户端0 训练后参数变化量：{param_change:.6f}")
            print(f"   正常范围：0.1~5；接近0则本地训练完全失效\n")

        return new_state, delta, self.num_samples, new_ctrl

# ===================== 工具函数 =====================
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total

def compute_communication_cost(state_dict):
    total_bytes = 0
    for param in state_dict.values():
        total_bytes += param.numel() * param.element_size()
    return total_bytes / (1024 * 1024)

def calibrate_bn(model, calib_loader, device):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()

    with torch.no_grad():
        for inputs, _ in calib_loader:
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs)

    model.eval()
    return model