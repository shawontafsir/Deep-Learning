# Define data augmentation transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from base_net import BaseNet
from res_net import ResNet

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(36, scale=(0.6, 1.0)),
    transforms.RandomCrop(32, padding=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 128

# Define the dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

# if __name__ == '__main__':
#     train_transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(20),
#         transforms.RandomResizedCrop(36, scale=(0.6, 1.0)),
#         transforms.RandomCrop(32, padding=5),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     batch_size = 128
#
#     # Define the dataset
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#
#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=train_transform)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
#
#     # # ----------------------------------------------------------------------
#     #
#     # Initialize the BaseNet model and define the loss and optimizer
#     base_net_20 = BaseNet(layers_count=6)
#     base_net_32 = BaseNet(layers_count=10)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(base_net_20.parameters(), lr=0.1, momentum=0.9)
#     #
#     # # Train the model
#     # for epoch in range(2):
#     #     running_loss = 0.0
#     #     for i, data in enumerate(trainloader, 0):
#     #     # for i in range(1):
#     #         inputs, labels = data
#     #         # inputs, labels = torch.from_numpy(np.float32(trainset.data)), torch.tensor(trainset.targets, dtype=torch.float32)
#     #
#     #         optimizer.zero_grad()
#     #
#     #         outputs = base_net_20(inputs)
#     #         loss = criterion(outputs, labels)
#     #         loss.backward()
#     #         optimizer.step()
#     #
#     #         running_loss += loss.item()
#     #         if i % 200 == 199:  # Print every 200 mini-batches
#     #             print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
#     #             running_loss = 0.0
#     #
#     # print("Finished Training")
#     #
#     # # Step 5: Evaluate the model on the test set
#     # correct = 0
#     # total = 0
#     # with torch.no_grad():
#     #     for data in testloader:
#     #         images, labels = data
#     #         outputs = base_net_20(images)
#     #         _, predicted = torch.max(outputs, 1)
#     #         total += labels.size(0)
#     #         correct += (predicted == labels).sum().item()
#     #
#     # print(f"Accuracy on the test set: {100 * correct / total}%")
#
#
#     # ------------------------------------------------------------------------
#
#     # Initialize the ResNet model and define the loss and optimizer
#     res_net_20 = ResNet(residual_blocks_count=3)
#     res_net_32 = ResNet(residual_blocks_count=5)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(res_net_20.parameters(), lr=0.01, momentum=0.9)
#
#     # Train the model
#     for epoch in range(1):
#         running_loss = 0.0
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data
#
#             optimizer.zero_grad()
#
#             outputs = res_net_20(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             if i % 20 == 19:  # Print every 200 mini-batches
#                 print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
#                 running_loss = 0.0
#
#     print("Finished Training")
#
#     # Step 5: Evaluate the model on the test set
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in testloader:
#             images, labels = data
#             outputs = res_net_20(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print(f"Accuracy on the test set: {100 * correct / total}%")
