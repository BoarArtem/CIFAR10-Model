import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_test_loader, get_train_loader

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # F(x)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # downsampling
        self.downsampling = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsampling = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.downsampling(x)

        return torch.relu(out)

class MNISTModel(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(MNISTModel, self).__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # layer part
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_resnet_model(num_classes, device):
    return MNISTModel(ResidualBlock, [2, 2, 2], num_classes).to(device)

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=0.001)

def get_criterion():
    return nn.CrossEntropyLoss()

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(num_epochs, model, optimizer, criterion, train_loader, device):
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()

        training_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            training_loss += loss.item()

        print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {(training_loss / len(train_loader)):.4f}")
        torch.save(model.state_dict(), "mnist_model.pth")

def test(test_loader, model, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = 100*(correct/total)

        print(f"Test accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    device = get_device()

    num_epochs = 10
    model = get_resnet_model(num_classes=10, device=device)
    optimizer = get_optimizer(model)
    criterion = get_criterion()

    train_loader = get_train_loader()
    test_loader = get_test_loader()

    train(num_epochs, model, optimizer, criterion, train_loader, device)
    test(test_loader, model, device)
