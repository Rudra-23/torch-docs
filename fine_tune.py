import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision

# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters
in_channels = 3
num_classes = 10

learning_rate = 0.001
batch_size = 64
num_epochs = 3


# Dataset
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = torchvision.models.vgg19(weights= "VGG19_Weights.DEFAULT")

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# Train Network
for epoch in range(1, num_epochs + 1):
    for batch_idx, (features, targets) in enumerate(train_loader):
       features = features.to(device)
       targets = targets.to(device)
       output = model(features)

       loss = criterion(output, targets)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       if (batch_idx + 1) % 100 == 0:
           print(f'Epoch: {epoch}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = scores.argmax(dim = 1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)