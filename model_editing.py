import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
from collections import defaultdict

class SparseSGDM(optim.SGD):
    """Sparse SGD with Momentum that accepts gradient masks"""
    def __init__(self, params, lr=0.01, momentum=0.9, gradient_masks=None):
        super(SparseSGDM, self).__init__(params, lr=lr, momentum=momentum)
        self.gradient_masks = gradient_masks or {}

    def set_gradient_masks(self, gradient_masks):
        """Set gradient masks for parameter updates"""
        self.gradient_masks = gradient_masks

    def step(self, closure=None):
        # Apply gradient masks before the update
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and id(p) in self.gradient_masks:
                    mask = self.gradient_masks[id(p)]
                    p.grad.data.mul_(mask)
        super(SparseSGDM, self).step(closure)

def compute_fisher_information(model, dataloader, device, num_samples=1000):
    """
    Compute Fisher Information Matrix diagonal elements
    Based on the code from https://github.com/iurada/talos-task-arithmetic
    """
    model.eval()
    fisher_info = defaultdict(float)
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_info[name] = torch.zeros_like(param.data)
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            sample_count += inputs.size(0)
    for name in fisher_info:
        fisher_info[name] /= sample_count
    return fisher_info

def calibrate_gradient_mask(model, dataloader, device, sparsity_ratio=0.5, 
                           num_calibration_rounds=3, num_samples_per_round=1000):
    """
    Calibrate gradient mask over multiple rounds
    Returns a dictionary mapping parameter IDs to binary masks
    """
    print(f"Calibrating gradient mask with {sparsity_ratio:.1%} sparsity over {num_calibration_rounds} rounds")
    cumulative_fisher = defaultdict(float)
    for round_idx in range(num_calibration_rounds):
        print(f"Calibration round {round_idx + 1}/{num_calibration_rounds}")
        fisher_info = compute_fisher_information(model, dataloader, device, num_samples_per_round)
        for name, fisher in fisher_info.items():
            cumulative_fisher[name] += fisher
    for name in cumulative_fisher:
        cumulative_fisher[name] /= num_calibration_rounds
    gradient_masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_values = cumulative_fisher[name].flatten()
            sorted_indices = torch.argsort(fisher_values)
            num_to_keep = int((1 - sparsity_ratio) * len(fisher_values))
            keep_indices = sorted_indices[:num_to_keep]
            mask = torch.zeros_like(param.data).flatten()
            mask[keep_indices] = 1.0
            mask = mask.reshape(param.data.shape)
            gradient_masks[id(param)] = mask
    print(f"Gradient mask created with {sparsity_ratio:.1%} sparsity")
    return gradient_masks

def calibrate_gradient_mask_alternative(model, dataloader, device, mask_type='least_sensitive',
                                       sparsity_ratio=0.5, num_calibration_rounds=3, 
                                       num_samples_per_round=1000):
    """
    Calibrate gradient mask with different selection strategies
    mask_type: 'least_sensitive', 'most_sensitive', 'lowest_magnitude', 'highest_magnitude', 'random'
    """
    print(f"Calibrating gradient mask using {mask_type} strategy with {sparsity_ratio:.1%} sparsity")
    if mask_type == 'random':
        gradient_masks = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                mask = (torch.rand_like(param.data) > sparsity_ratio).float()
                gradient_masks[id(param)] = mask
        return gradient_masks
    cumulative_fisher = defaultdict(float)
    for round_idx in range(num_calibration_rounds):
        print(f"Calibration round {round_idx + 1}/{num_calibration_rounds}")
        fisher_info = compute_fisher_information(model, dataloader, device, num_samples_per_round)
        for name, fisher in fisher_info.items():
            cumulative_fisher[name] += fisher
    for name in cumulative_fisher:
        cumulative_fisher[name] /= num_calibration_rounds
    gradient_masks = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            if mask_type in ['least_sensitive', 'most_sensitive']:
                fisher_values = cumulative_fisher[name].flatten()
                sorted_indices = torch.argsort(fisher_values)
                if mask_type == 'least_sensitive':
                    keep_indices = sorted_indices[:int((1 - sparsity_ratio) * len(fisher_values))]
                else:
                    keep_indices = sorted_indices[-int((1 - sparsity_ratio) * len(fisher_values)):]
            elif mask_type in ['lowest_magnitude', 'highest_magnitude']:
                param_values = torch.abs(param.data).flatten()
                sorted_indices = torch.argsort(param_values)
                if mask_type == 'lowest_magnitude':
                    keep_indices = sorted_indices[:int((1 - sparsity_ratio) * len(param_values))]
                else:
                    keep_indices = sorted_indices[-int((1 - sparsity_ratio) * len(param_values)):]
            mask = torch.zeros_like(param.data).flatten()
            mask[keep_indices] = 1.0
            mask = mask.reshape(param.data.shape)
            gradient_masks[id(param)] = mask
    print(f"Gradient mask created using {mask_type} strategy with {sparsity_ratio:.1%} sparsity")
    return gradient_masks 