import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets  import CIFAR10
from torch.utils.data import DataLoader

transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465,), (0.247, 0.243, 0.261,))])

train_dataset = CIFAR10(root="/data", train=True, transform=transforms, download=True)
test_dataset = CIFAR10(root="/data", train=False)

def get_train_loader(batch_size = 32, shuffle = True):
    return DataLoader(train_dataset, batch_size, shuffle)

def get_test_loader(batch_size = 32, shuffle = False):
    return DataLoader(test_dataset, batch_size, shuffle)


for X_batch, y_batch in get_train_loader():
    print(X_batch.shape)
    print(y_batch.shape)

    break