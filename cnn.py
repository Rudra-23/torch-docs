import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Model
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # shape = (batch_sise, in_channels, height, width)
        self.conv1 = nn.Conv2d(in_channels, 30, kernel_size=(3, 3), stride = (1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(30, 15, kernel_size=(3, 3), stride= (1, 1), padding=(1, 1))
 
        self.fc1 = nn.Linear(7*7*15, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters
in_channels = 1
num_classes = 10

learning_rate = 0.001
batch_size = 64
num_epochs = 1


# Dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

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