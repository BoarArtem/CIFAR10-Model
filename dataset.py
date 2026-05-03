import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = FashionMNIST("/data", train=True, download=True, transform=transforms)
test_dataset = FashionMNIST("/data", train=False, download=False)

def get_train_loader(batch_size = 32, shuffle = True):
    return DataLoader(train_dataset, batch_size, shuffle)

def get_test_loader(batch_size = 32, shuffle = False):
    return DataLoader(train_dataset, batch_size, shuffle)

train_loader = get_train_loader()

for inputs, labels in train_loader:
    print(inputs.shape)
    print(labels.shape)

    break
