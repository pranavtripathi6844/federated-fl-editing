import torch
import torchvision
from torch.utils.data import Subset, DataLoader
import pickle

# Load validation indices
with open('val_indices.pkl', 'rb') as f:
    val_indices = pickle.load(f)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
])

dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
val_set = Subset(dataset, val_indices)
val_loader = DataLoader(val_set, batch_size=64)

# Load model
model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
embed_dim = model.embed_dim if hasattr(model, 'embed_dim') else 384
model.head = torch.nn.Linear(embed_dim, 100)
model.load_state_dict(torch.load('model_checkpoint.pth'))  # Use your checkpoint filename
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Evaluation
correct = 0
total = 0
val_loss = 0.0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        val_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    val_loss /= total
    val_acc = correct / total

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")