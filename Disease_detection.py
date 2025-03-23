import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd

# Dataset transformation
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("Dataset", transform=transform)

# Splitting data
indices = list(range(len(dataset)))
np.random.shuffle(indices)
split = int(np.floor(0.85 * len(dataset)))
validation = int(np.floor(0.70 * split))
train_indices, validation_indices, test_indices = (
    indices[:validation], indices[validation:split], indices[split:]
)

train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)
test_sampler = SubsetRandomSampler(test_indices)

targets_size = len(dataset.class_to_idx)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2)
        )
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K)
        )
    
    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)
        out = self.dense_layers(out)
        return out

# Model initialization
model = CNN(targets_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# Data Loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler)

# Training function
def train(model, criterion, train_loader, validation_loader, epochs):
    for e in range(epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                validation_loss += criterion(model(inputs), targets).item()
        
        print(f"Epoch {e+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.3f} - Validation Loss: {validation_loss/len(validation_loader):.3f}")

# Training
epochs = 5
train(model, criterion, train_loader, validation_loader, epochs)

# Save Model
torch.save(model.state_dict(), 'plant_disease_model.pt')

# Load Model
model.load_state_dict(torch.load("plant_disease_model.pt"))
model.eval()

# Accuracy Function
def accuracy(loader):
    n_correct, n_total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, predictions = torch.max(model(inputs), 1)
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]
    return n_correct / n_total

print(f"Train Accuracy: {accuracy(train_loader):.3f}")
print(f"Test Accuracy: {accuracy(test_loader):.3f}")
print(f"Validation Accuracy: {accuracy(validation_loader):.3f}")

# Single Image Prediction
data = pd.read_csv("disease_info.csv", encoding="cp1252")
class_map = {v: k for k, v in dataset.class_to_idx.items()}

def single_prediction(image_path):
    image = Image.open(image_path).resize((224, 224))
    input_data = TF.to_tensor(image).view((-1, 3, 224, 224))
    output = model(input_data).detach().numpy()
    index = np.argmax(output)
    print(f"Prediction: {data['disease_name'][index]}")

# Example Prediction
single_prediction("test_images/Apple_ceder_apple_rust.JPG")
