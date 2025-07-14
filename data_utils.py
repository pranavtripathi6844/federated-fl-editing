import torch
import numpy as np
from torch.utils.data import Dataset, Subset, random_split
import torchvision
import torchvision.transforms as transforms
from collections import defaultdict
import random

def create_cifar100_datasets(root='./data', image_size=224):
    """Create CIFAR-100 datasets with proper train/val/test splits"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])
    
    # Load datasets
    full_trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform)
    
    # Split training data into train and validation
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    
    return trainset, valset, testset

def create_iid_split(dataset, num_clients, samples_per_client=None):
    """
    Create IID split of dataset among clients
    Each client gets approximately equal number of samples uniformly distributed over class labels
    """
    if samples_per_client is None:
        samples_per_client = len(dataset) // num_clients
    
    # Shuffle indices
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Split indices among clients
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(dataset)
        client_indices = indices[start_idx:end_idx]
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets

def create_non_iid_split(dataset, num_clients, num_classes_per_client, samples_per_client=None):
    """
    Create non-IID split of dataset among clients
    Each client gets samples from only num_classes_per_client classes
    """
    if samples_per_client is None:
        samples_per_client = len(dataset) // num_clients
    
    # Get class labels for each sample
    class_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_labels.append(label)
    
    # Group samples by class
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(class_labels):
        class_to_indices[label].append(idx)
    
    # Create client datasets
    client_datasets = []
    available_classes = list(range(100))  # CIFAR-100 has 100 classes
    
    for client_id in range(num_clients):
        # Randomly select classes for this client
        if len(available_classes) < num_classes_per_client:
            # If not enough classes left, reuse all classes
            client_classes = random.sample(list(range(100)), num_classes_per_client)
        else:
            client_classes = random.sample(available_classes, num_classes_per_client)
        
        # Collect indices for selected classes
        client_indices = []
        for cls in client_classes:
            client_indices.extend(class_to_indices[cls])
        
        # Randomly sample from available indices
        if len(client_indices) > samples_per_client:
            client_indices = random.sample(client_indices, samples_per_client)
        
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets

def get_data_statistics(client_datasets, num_classes=100):
    """Analyze the distribution of classes across clients"""
    stats = []
    for i, dataset in enumerate(client_datasets):
        class_counts = defaultdict(int)
        for idx in dataset.indices:
            _, label = dataset.dataset[idx]
            class_counts[label] += 1
        
        stats.append({
            'client_id': i,
            'num_samples': len(dataset),
            'num_classes': len(class_counts),
            'class_distribution': dict(class_counts)
        })
    
    return stats

def print_data_statistics(stats):
    """Print statistics about data distribution"""
    print(f"Data distribution across {len(stats)} clients:")
    for stat in stats:
        print(f"Client {stat['client_id']}: {stat['num_samples']} samples, {stat['num_classes']} classes")
    
    # Calculate heterogeneity metrics
    class_counts = [stat['num_classes'] for stat in stats]
    print(f"Average classes per client: {np.mean(class_counts):.2f}")
    print(f"Std classes per client: {np.std(class_counts):.2f}")
    print(f"Min classes per client: {min(class_counts)}")
    print(f"Max classes per client: {max(class_counts)}") 