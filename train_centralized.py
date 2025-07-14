import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from vision_transformer import vit_small, vit_tiny
import time
import os
import math
import argparse
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
from model_editing import SparseSGDM, calibrate_gradient_mask_alternative

# Set device based on LOCAL_RANK for distributed training
try:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
except Exception:
    local_rank = 0
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

def parse_args():
    """Parse command line arguments with optimizations for RTX 2060."""
    parser = argparse.ArgumentParser(description='Centralized Training with DINO ViT')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='vit_small', 
                       choices=['vit_tiny', 'vit_small'],
                       help='Model architecture (default: vit_small)')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for ViT (default: 16)')
    
    # Data configuration
    parser.add_argument('--image_size', type=int, default=128,
                       help='Input image size (default: 128 for speed)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256 for RTX 2060)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers (default: 4)')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience (default: 15)')
    
    # Optimization
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping value (default: 1.0)')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='TensorBoard log directory (default: ./logs)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint directory (default: ./checkpoints)')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--force_reset', action='store_true',
                       help='Force reset training (ignore existing checkpoints)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cuda/cpu)')
    
    # Model editing arguments
    parser.add_argument('--use_model_editing', action='store_true', help='Enable model editing (sparse gradient masking)')
    parser.add_argument('--mask_type', type=str, default='least_sensitive', choices=['least_sensitive', 'most_sensitive', 'lowest_magnitude', 'highest_magnitude', 'random'], help='Mask type for model editing')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity ratio for gradient mask (default: 0.5)')
    parser.add_argument('--num_calibration_rounds', type=int, default=3, help='Number of calibration rounds for mask')
    parser.add_argument('--num_samples_per_round', type=int, default=1000, help='Samples per calibration round')
    parser.add_argument('--mask_recalibration_freq', type=int, default=10, help='Recalibrate mask every N epochs')
    
    return parser.parse_args()

def setup_device(args) -> torch.device:
    """Setup device based on args and available hardware."""
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Print GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if local_rank == 0:
                print(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = torch.device('cpu')
            if local_rank == 0:
                print("CUDA not available, using CPU")
    else:
        device = torch.device(args.device)
    
    if local_rank == 0:
        print(f"Device: {device}")
    return device

def create_model(args) -> nn.Module:
    """Create model based on arguments."""
    if args.model == 'vit_tiny':
        model = vit_tiny(patch_size=args.patch_size, num_classes=100)
        if local_rank == 0:
            print("Using ViT-Tiny model")
    else:
        model = vit_small(patch_size=args.patch_size, num_classes=100)
        if local_rank == 0:
            print("Using ViT-Small model")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if local_rank == 0:
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model

def create_transforms(args) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create training and validation transforms."""
    # CIFAR-100 normalization
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408], 
        std=[0.2675, 0.2565, 0.2761]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform

def load_data(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load and prepare CIFAR-100 data."""
    print("Loading CIFAR-100 dataset...")
    
    train_transform, val_transform = create_transforms(args)
    
    # Load full training set
    full_trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    # Load test set
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=val_transform
    )
    
    # Split training data
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    # Update validation set transform
    valset.dataset.transform = val_transform
    
    print(f"Data split: {len(trainset)} train, {len(valset)} val, {len(testset)} test")
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    valloader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return trainloader, valloader, testloader

def train_epoch(model, trainloader, criterion, optimizer, scaler, device, args):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if args.mixed_precision:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress update every 50 batches
        if batch_idx % 50 == 0 and local_rank == 0:
            print(f"Batch {batch_idx}/{len(trainloader)}, Loss: {loss.item():.4f}")
    
    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    
    return train_loss, train_acc

def validate(model, valloader, criterion, device, args):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if args.mixed_precision:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(valloader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, 
                   patience_counter, args, is_best=False):
    """Save model checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'model_config': {
            'model': args.model,
            'patch_size': args.patch_size,
            'image_size': args.image_size
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'centralized_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_path)
        if local_rank == 0:
            print(f"Best model saved: {best_path}")

def load_checkpoint(model, optimizer, scheduler, args):
    """Load checkpoint if exists and compatible."""
    checkpoint_path = os.path.join(args.checkpoint_dir, 'centralized_checkpoint.pth')
    
    if args.force_reset:
        if local_rank == 0:
            print("Force reset enabled, ignoring existing checkpoints")
        return 0, math.inf, 0
    
    if os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if model configuration matches
            if 'model_config' in checkpoint:
                saved_config = checkpoint['model_config']
                current_config = {
                    'model': args.model,
                    'patch_size': args.patch_size,
                    'image_size': args.image_size
                }
                
                if saved_config != current_config:
                    if local_rank == 0:
                        print(f"Model configuration mismatch!")
                        print(f"Saved: {saved_config}")
                        print(f"Current: {current_config}")
                        print("Starting from scratch due to configuration change")
                    return 0, math.inf, 0
            
            # Try to load model state dict
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', math.inf)
                patience_counter = checkpoint.get('patience_counter', 0)
                
                if local_rank == 0:
                    print(f"Resumed from epoch {start_epoch} with best val loss {best_val_loss:.4f}")
                return start_epoch, best_val_loss, patience_counter
                
            except RuntimeError as e:
                if local_rank == 0:
                    print(f"Model state dict mismatch: {e}")
                    print("Starting from scratch due to model architecture change")
                return 0, math.inf, 0
                
        except Exception as e:
            if local_rank == 0:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch")
            return 0, math.inf, 0
    else:
        if local_rank == 0:
            print("No checkpoint found, starting from scratch")
        return 0, math.inf, 0

def main():
    """Main training function."""
    args = parse_args()
    
    # Setup device (override with distributed device)
    # device is already set above for distributed training
    
    # Create model
    model = create_model(args)
    model.to(device)
    
    # Load data
    trainloader, valloader, testloader = load_data(args)
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if args.mixed_precision else None
    
    # Model editing logic
    if args.use_model_editing:
        print("[Model Editing] Using SparseSGDM optimizer and gradient mask calibration.")
        optimizer = SparseSGDM(model.parameters(), lr=args.lr, momentum=args.momentum)
        # Initial mask calibration
        print(f"[Model Editing] Calibrating gradient mask (type={args.mask_type}, sparsity={args.sparsity_ratio})...")
        gradient_masks = calibrate_gradient_mask_alternative(
            model, trainloader, device,
            mask_type=args.mask_type,
            sparsity_ratio=args.sparsity_ratio,
            num_calibration_rounds=args.num_calibration_rounds,
            num_samples_per_round=args.num_samples_per_round
        )
        optimizer.set_gradient_masks(gradient_masks)
    else:
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    # Load checkpoint if exists
    start_epoch, best_val_loss, patience_counter = load_checkpoint(
        model, optimizer, scheduler, args
    )
    
    # Training loop
    if local_rank == 0:
        print(f"Starting training for {args.epochs} epochs from epoch {start_epoch}")
    total_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Recalibrate mask if needed
        if args.use_model_editing and (epoch == 0 or (epoch % args.mask_recalibration_freq == 0)):
            print(f"[Model Editing] Recalibrating gradient mask at epoch {epoch+1}...")
            gradient_masks = calibrate_gradient_mask_alternative(
                model, trainloader, device,
                mask_type=args.mask_type,
                sparsity_ratio=args.sparsity_ratio,
                num_calibration_rounds=args.num_calibration_rounds,
                num_samples_per_round=args.num_samples_per_round
            )
            optimizer.set_gradient_masks(gradient_masks)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, scaler, device, args
        )
        
        # Validate
        val_loss, val_acc = validate(model, valloader, criterion, device, args)
        
        # Update scheduler
        scheduler.step()
        
        # Logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], epoch)
        
        epoch_time = time.time() - epoch_start_time
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                  f"Time: {epoch_time/60:.2f}min")
        
        # Early stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss, 
                patience_counter, args, is_best
            )
        
        # Early stopping check
        if patience_counter >= args.patience and local_rank == 0:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation
    if local_rank == 0:
        print("Evaluating on test set...")
    test_loss, test_acc = validate(model, testloader, criterion, device, args)
    if local_rank == 0:
        print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'centralized_model_final.pth')
    torch.save(model.state_dict(), final_path)
    if local_rank == 0:
        print(f"Final model saved: {final_path}")
    
    # Training summary
    total_time = time.time() - total_start_time
    if local_rank == 0:
        print(f"Training completed in {total_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if local_rank == 0:
        print(f"Final test accuracy: {test_acc:.2f}%")
    
    writer.close()

if __name__ == '__main__':
    import random
    import numpy as np
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main()
