import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_train_loader


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # (batch_size, channel, height, width)
        # (32, 3, 32, 32)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            # feature - (32 - 3 + 2)/1 + 1 = 32
            # (32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32 / 2 = 16
            # (32, 16, 16)


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 16 / 2 = 8
            # (64, 8, 8)
        )

        self.fc_layer = nn.Sequential(
            # Linear - (batch_size, features)
            # linear input - 64 * 8 * 8 = 4096
            nn.Flatten(),

            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_cifar_model():
    return CIFAR10Model().to(device)

def get_optimization(model):
    return optim.Adam(model.parameters(), lr=0.001)

def get_criterion():
    return nn.CrossEntropyLoss()


def train(model, optim, criterion, train_loader, num_epochs = 10):
    print("Starting training...")
    for epoch in range(num_epochs):
        training_loss = 0
        correct = 0
        total = 0

        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optim.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optim.step()

            training_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {(training_loss / len(train_loader)):.4f}, Accuracy: {(correct/total)*100:.4f}")
        torch.save(model.state_dict(), "cifar10_model.pth")

if __name__ == "__main__":
    model = get_cifar_model()
    optim = get_optimization(model)
    criterion = get_criterion()
    train_loader = get_train_loader()

    train(model, optim, criterion, train_loader)
