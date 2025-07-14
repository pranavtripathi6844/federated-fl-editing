import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import time
import os
import math
import argparse
import json

import torchvision
import torchvision.transforms as transforms

from data_utils import create_cifar100_datasets
from federated_learning_mps import FederatedLearning

import optuna
import subprocess

try:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
except Exception:
    local_rank = 0

print("Script started", flush=True)

def main():
    parser = argparse.ArgumentParser(description='Federated Learning with Model Editing')
    parser.add_argument('--image_size', type=int, default=128, help='Image size for training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--model_type', type=str, default='vit_tiny', 
                       choices=['vit_tiny', 'vit_small'], help='Model architecture')
    parser.add_argument('--num_clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--client_fraction', type=float, default=0.1, help='Fraction of clients participating per round')
    parser.add_argument('--num_rounds', type=int, default=100, help='Number of communication rounds')
    parser.add_argument('--local_epochs', type=int, default=4, help='Number of local epochs per client')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--data_distribution', type=str, default='iid', 
                       choices=['iid', 'non_iid'], help='Data distribution among clients')
    parser.add_argument('--num_classes_per_client', type=int, default=5, 
                       help='Number of classes per client (for non-iid)')
    parser.add_argument('--use_model_editing', action='store_true', 
                       help='Use model editing with gradient masking')
    parser.add_argument('--mask_type', type=str, default='least_sensitive',
                       choices=['least_sensitive', 'most_sensitive', 'lowest_magnitude', 
                               'highest_magnitude', 'random'], help='Gradient mask selection strategy')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, 
                       help='Sparsity ratio for gradient masking')
    parser.add_argument('--num_calibration_rounds', type=int, default=3, 
                       help='Number of calibration rounds for gradient masks')
    parser.add_argument('--num_samples_per_round', type=int, default=1000, 
                       help='Number of samples per calibration round')
    parser.add_argument('--mask_recalibration_freq', type=int, default=10, 
                       help='Frequency of gradient mask recalibration')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs_fl', help='Log directory')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency')
    parser.add_argument('--force_reset', action='store_true', help='Force reset: clear log_dir and checkpoints before training')
    args = parser.parse_args()
    config = {
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'model_type': args.model_type,
        'num_clients': args.num_clients,
        'client_fraction': args.client_fraction,
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'learning_rate': args.learning_rate,
        'data_distribution': args.data_distribution,
        'num_classes_per_client': args.num_classes_per_client,
        'use_model_editing': args.use_model_editing,
        'mask_type': args.mask_type,
        'sparsity_ratio': args.sparsity_ratio,
        'num_calibration_rounds': args.num_calibration_rounds,
        'num_samples_per_round': args.num_samples_per_round,
        'mask_recalibration_freq': args.mask_recalibration_freq,
        'seed': args.seed,
        'log_dir': args.log_dir,
        'checkpoint_freq': args.checkpoint_freq,
        'best_model_path': f'best_model_fl_{args.data_distribution}.pth',
        'checkpoint_path': f'checkpoint_fl_{args.data_distribution}.pth'
    }
    # Handle force_reset: clear log_dir and checkpoints
    if args.force_reset:
        import shutil
        if os.path.exists(args.log_dir):
            print(f"[force_reset] Removing log_dir: {args.log_dir}")
            shutil.rmtree(args.log_dir)
        # Remove checkpoints if present
        for fname in [f'best_model_fl_{args.data_distribution}.pth', f'checkpoint_fl_{args.data_distribution}.pth']:
            if os.path.exists(fname):
                print(f"[force_reset] Removing checkpoint: {fname}")
                os.remove(fname)
    os.makedirs(args.log_dir, exist_ok=True)
    with open(os.path.join(args.log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    if local_rank == 0:
        print("Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("\nLoading CIFAR-100 datasets...")
    trainset, valset, testset = create_cifar100_datasets(
        root='./data', 
        image_size=args.image_size
    )
    if local_rank == 0:
        print(f"Data loaded: {len(trainset)} training samples, {len(valset)} validation samples, {len(testset)} test samples")
    try:
        fl = FederatedLearning(config)
        global_model, best_acc = fl.run_federated_training(trainset, valset, testset)
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        best_acc = 0.0
    if local_rank == 0:
        print(f"\nFinal Results:")
        print(f"Best test accuracy: {best_acc:.2f}%")
        print(f"Configuration saved to: {os.path.join(args.log_dir, 'config.json')}")
        print(f"Logs saved to: {args.log_dir}")
    # Write best accuracy to file for Optuna
    with open(os.path.join(args.log_dir, 'best_acc.txt'), 'w') as f:
        f.write(str(best_acc))

# Optuna objective for hyperparameter search
# Always uses vit_small for fair comparison with centralized
# Reads best_acc.txt for robust metric extraction

def optuna_objective(trial):
    learning_rate = trial.suggest_categorical('learning_rate', [0.1, 0.05, 0.01, 0.005, 0.001])
    client_fraction = trial.suggest_categorical('client_fraction', [0.1, 0.3, 0.5, 1.0])
    local_epochs = trial.suggest_categorical('local_epochs', [1, 2, 5, 10])
    num_rounds = trial.suggest_categorical('num_rounds', [50, 100, 200])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
    log_dir = f"./logs/optuna_lr{learning_rate}_cf{client_fraction}_le{local_epochs}_nr{num_rounds}_bs{batch_size}"
    cmd = [
        "python", "train_federated.py",
        "--model_type", "vit_small",
        "--image_size", "128",
        "--batch_size", str(batch_size),
        "--num_clients", "10",
        "--client_fraction", str(client_fraction),
        "--num_rounds", str(num_rounds),
        "--local_epochs", str(local_epochs),
        "--learning_rate", str(learning_rate),
        "--data_distribution", "non_iid",
        "--log_dir", log_dir,
        "--force_reset"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if local_rank == 0:
        print(f"[Optuna] Trial log_dir: {log_dir}")
    if result.returncode != 0:
        if local_rank == 0:
            print(f"[Optuna] Subprocess failed. stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    best_acc = 0.0
    try:
        with open(f"{log_dir}/best_acc.txt") as f:
            best_acc = float(f.read())
        if local_rank == 0:
            print(f"[Optuna] Read best_acc.txt: {best_acc}")
    except Exception as e:
        if local_rank == 0:
            print(f"[Optuna] Could not read best_acc.txt: {e}")
    return best_acc

if __name__ == '__main__':
    main() 