# prepare_data.py
import torchvision
from torchvision import transforms
from torch.utils.data import random_split

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
val_size = 5000
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])

# Save indices for reproducibility
import pickle
with open('train_indices.pkl', 'wb') as f:
    pickle.dump(train_set.indices, f)
with open('val_indices.pkl', 'wb') as f:
    pickle.dump(val_set.indices, f)