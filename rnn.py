import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size * sequence_length, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # out, _ = self.rnn(x, h0)
        out, _ = self.gru(x, h0)

        out = out.reshape(out.shape[0], -1)
        out = self.fc1(out)

        return out


# Device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters
input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 2
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
model = RNN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes, num_layers=num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


# Train Network
for epoch in range(1, num_epochs + 1):
    for batch_idx, (features, targets) in enumerate(train_loader):
       features = features.squeeze(1).to(device)
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            predictions = scores.argmax(dim = 1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}")

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)