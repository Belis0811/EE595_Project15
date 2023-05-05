import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class PhysicsLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, interaction_strength=0.5):
        super(PhysicsLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.interaction_strength = interaction_strength
        self.neighbor_forces = nn.Parameter(torch.zeros(out_channels) * 2.4)

    def forward(self, x):
        # Compute the convolution
        y = self.conv(x)

        # Compute the forces on the neighboring layers
        forces = torch.cat([self.neighbor_forces[-1:], self.neighbor_forces[:-1]], dim=0) + torch.cat(
            [self.neighbor_forces[1:], self.neighbor_forces[:1]], dim=0)
        forces = forces.view(1, -1, 1, 1).expand_as(y)
        # Apply the forces to the neighboring layers
        y = y + self.interaction_strength * forces
        return y


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            PhysicsLayer(3, 64, kernel_size=11, stride=4, padding=2, interaction_strength=2.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            PhysicsLayer(64, 192, kernel_size=5, padding=2, interaction_strength=2.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            PhysicsLayer(192, 384, kernel_size=3, padding=1, interaction_strength=2.8),
            nn.ReLU(inplace=True),
            PhysicsLayer(384, 256, kernel_size=3, padding=1, interaction_strength=2.8),
            nn.ReLU(inplace=True),
            PhysicsLayer(256, 256, kernel_size=3, padding=1, interaction_strength=2.8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    # Define the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms for data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./cifar', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./cifar', train=False, download=False, transform=transform_test)

    # Define the data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Define the model
    model = AlexNet(num_classes=10).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Define the scheduler
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.001)

    # Train the model
    num_epochs = 50

    train_loss = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100.0 * correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc / 100.0)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%")

        # Update the scheduler
    # scheduler.step()

    # Test the model
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().item()

    test_loss /= len(test_dataset)
    test_acc = 100.0 * correct / total
    for i in range(50):
        test_losses.append(test_loss)
        test_accuracy.append(test_acc / 100.0)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    epochs = range(1, 51)

    # plot training and validation loss
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # plot training and validation accuracy
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
