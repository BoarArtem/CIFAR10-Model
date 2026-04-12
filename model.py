import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, test_loader


class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )

        self.fc_layer = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10Model().to(device)

criterion = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=0.001)

num_epoch = 10

if __name__ == "__main__":
    for epoch in range(num_epoch):
        training_loss = 0

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

        print(f"Epoch: {epoch + 1}/{num_epoch}, Loss: {training_loss}")
        torch.save(model.state_dict(), "cifar_model.pth")

    correct = 0
    total = 0

    with torch.no_grad():
        model.eval()

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        print(f"Accuracy: {100 * (correct / total)}%")
        # after training, I got 74% accuracy