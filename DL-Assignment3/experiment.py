"""
Experiment 1: Compare the performance on layering a model
1. Compare the training and testing accuracy over epoch for BaseNet-20 and BaseNet-32.
2. Compare the training and testing loss over epoch for ResNet-20 and ResNet-32.

Experiment 2: Compare the performance of BaseNet and ResNet
1. Compare the training and testing accuracy over epoch for BaseNet-20 and ResNet-20.
2. Compare the training and testing accuracy over epoch for BaseNet-32 and ResNet-32.
3.  Compare the training and testing loss over epoch for BaseNet-20 and ResNet-20.
4. Compare the training and testing loss over epoch for BaseNet-32 and ResNet-32.

Experiment 3: How different optimizer affect the convergence?
For BaseNet-32 and ResNet-32 model using different optimizer: learning schedule, momentum and adam.
1. Compare the training accuracy over epochs.
2. Compare the training loss over epochs.
"""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR

from base_net import BaseNet
from res_net import ResNet
from data_preparation import trainloader, testloader
from data_plotting import PlotGeneration


if __name__ == "__main__":
    # Initialize the BaseNet models and define the loss and optimizer along with schedular
    base_net_20 = BaseNet(layers_count=6)
    base_net_20_optimizer = optim.SGD(base_net_20.parameters(), lr=0.1)
    base_net_20_optimizer_schedular = StepLR(base_net_20_optimizer, step_size=200, gamma=0.1)  # For each 200 epoch

    base_net_32 = BaseNet(layers_count=10)
    base_net_32_optimizer = optim.SGD(base_net_32.parameters(), lr=0.1)
    base_net_32_optimizer_schedular = StepLR(base_net_32_optimizer, step_size=200, gamma=0.1)  # For each 200 epoch

    base_net_32_adam_optimizer = optim.Adam(base_net_32.parameters(), lr=0.1, betas=(0.9, 0.99))   # using commonly used momentum 0.9
    base_net_32_adam_optimizer_schedular = StepLR(base_net_32_adam_optimizer, step_size=200, gamma=0.1)   # For each 200 epoch

    # Initialize the ResNet models and define the loss and optimizer
    res_net_20 = ResNet(residual_blocks_count=3)
    res_net_20_optimizer = optim.SGD(res_net_20.parameters(), lr=0.1)
    res_net_20_optimizer_schedular = StepLR(res_net_20_optimizer, step_size=200, gamma=0.1)  # For each 200 epoch

    res_net_32 = ResNet(residual_blocks_count=5)
    res_net_32_optimizer = optim.SGD(res_net_32.parameters(), lr=0.1)
    res_net_32_optimizer_schedular = StepLR(res_net_32_optimizer, step_size=200, gamma=0.1)  # For each 200 epoch

    res_net_32_adam_optimizer = optim.Adam(res_net_32.parameters(), lr=0.1, betas=(0.9, 0.99))  # using commonly used momentum 0.9
    res_net_32_adam_optimizer_schedular = StepLR(res_net_32_adam_optimizer, step_size=200, gamma=0.1)  # For each 200 epoch

    epochs = 500
    criterion = nn.CrossEntropyLoss()   # For classification

    # Relevant list of results per epoch
    base_net_20_epoch_training_losses, base_net_32_epoch_training_losses = [], []
    base_net_20_epoch_testing_losses, base_net_32_epoch_testing_losses = [], []
    base_net_20_epoch_training_accuracy, base_net_32_epoch_training_accuracy = [], []
    base_net_20_epoch_testing_accuracy, base_net_32_epoch_testing_accuracy = [], []
    base_net_32_training_accuracy_for_adam, base_net_32_training_losses_for_adam = [], []

    res_net_20_epoch_training_losses, res_net_32_epoch_training_losses = [], []
    res_net_20_epoch_testing_losses, res_net_32_epoch_testing_losses = [], []
    res_net_20_epoch_training_accuracy, res_net_32_epoch_training_accuracy = [], []
    res_net_20_epoch_testing_accuracy, res_net_32_epoch_testing_accuracy = [], []
    res_net_32_training_accuracy_for_adam, res_net_32_training_losses_for_adam = [], []

    # --------------------- Training Start --------------------------------------------------------
    for epoch in range(epochs):
        # initialization for training experiments
        base_net_20_epoch_training_loss, base_net_32_epoch_training_loss = 0, 0
        base_net_20_epoch_training_accurate, base_net_32_epoch_training_accurate = 0, 0
        base_net_32_epoch_training_accurate_for_adam, base_net_32_epoch_training_loss_for_adam = 0, 0

        res_net_20_epoch_training_loss, res_net_32_epoch_training_loss = 0, 0
        res_net_20_epoch_training_accurate, res_net_32_epoch_training_accurate = 0, 0
        res_net_32_epoch_training_accurate_for_adam, res_net_32_epoch_training_loss_for_adam = 0, 0
        total_train_data = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            total_train_data += len(labels)

            # BaseNet-20 Training
            base_net_20_optimizer.zero_grad()
            base_net_20_outputs = base_net_20(inputs)

            _, predicted_20 = torch.max(base_net_20_outputs, 1)   # get predicted labels: torch.max returns (values, labels)
            base_net_20_epoch_training_accurate += (predicted_20 == labels).sum().item()

            base_net_20_loss = criterion(base_net_20_outputs, labels)
            base_net_20_loss.backward()
            base_net_20_optimizer.step()
            base_net_20_epoch_training_loss += base_net_20_loss.item()

            # Print every 200 mini-batches
            if (i+1) % 200 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {base_net_20_epoch_training_loss / 200:.3f}")

            # BaseNet-32 Training
            base_net_32_optimizer.zero_grad()
            base_net_32_outputs = base_net_32(inputs)

            _, predicted_32 = torch.max(base_net_32_outputs, 1)   # get predicted labels: torch.max returns (values, labels)
            base_net_32_epoch_training_accurate += (predicted_32 == labels).sum().item()

            base_net_32_loss = criterion(base_net_32_outputs, labels)
            base_net_32_loss.backward()
            base_net_32_optimizer.step()
            base_net_32_epoch_training_loss += base_net_32_loss.item()

            # Print every 200 mini-batches
            if (i + 1) % 200 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {base_net_32_epoch_training_loss / 200:.3f}")

            # ResNet-20 Training
            res_net_20_optimizer.zero_grad()
            res_net_20_outputs = res_net_20(inputs)

            _, predicted_20 = torch.max(res_net_20_outputs, 1)  # get predicted labels: torch.max returns (values, labels)
            res_net_20_epoch_training_accurate += (predicted_20 == labels).sum().item()

            res_net_20_loss = criterion(res_net_20_outputs, labels)
            res_net_20_loss.backward()
            res_net_20_optimizer.step()
            res_net_20_epoch_training_loss += res_net_20_loss.item()

            # Print every 200 mini-batches
            if (i+1) % 200 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {res_net_20_epoch_training_loss / 200:.3f}")

            # ResNet-32 Training
            res_net_32_optimizer.zero_grad()
            res_net_32_outputs = res_net_32(inputs)

            _, predicted_32 = torch.max(res_net_32_outputs, 1)  # get predicted labels: torch.max returns (values, labels)
            res_net_32_epoch_training_accurate += (predicted_32 == labels).sum().item()

            res_net_32_loss = criterion(res_net_32_outputs, labels)
            res_net_32_loss.backward()
            res_net_32_optimizer.step()
            res_net_32_epoch_training_loss += res_net_32_loss.item()

            # Print every 200 mini-batches
            if (i + 1) % 200 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {res_net_32_epoch_training_loss / 200:.3f}")

            # BaseNet-32 Training with adam optimizer
            base_net_32_adam_optimizer.zero_grad()
            base_net_32_outputs_adam = base_net_32(inputs)

            _, predicted_32 = torch.max(base_net_32_outputs, 1)  # get predicted labels: torch.max returns (values, labels)
            base_net_32_epoch_training_accurate_for_adam += (predicted_32 == labels).sum().item()

            base_net_32_loss_adam = criterion(base_net_32_outputs_adam, labels)
            base_net_32_loss_adam.backward()
            base_net_32_adam_optimizer.step()
            base_net_32_epoch_training_loss_for_adam += base_net_32_loss_adam.item()

            # Print every 200 mini-batches
            if (i + 1) % 200 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {base_net_32_epoch_training_loss_for_adam / 200:.3f}")

            # ResNet-32 Training with adam optimizer
            res_net_32_adam_optimizer.zero_grad()
            res_net_32_outputs_adam = res_net_32(inputs)

            _, predicted_32 = torch.max(res_net_32_outputs_adam, 1)  # get predicted labels: torch.max returns (values, labels)
            res_net_32_epoch_training_accurate_for_adam += (predicted_32 == labels).sum().item()

            res_net_32_loss_adam = criterion(res_net_32_outputs_adam, labels)
            res_net_32_loss_adam.backward()
            res_net_32_adam_optimizer.step()
            res_net_32_epoch_training_loss_for_adam += res_net_32_loss_adam.item()

            # Print every 200 mini-batches
            if (i + 1) % 200 == 0:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {res_net_32_epoch_training_loss_for_adam / 200:.3f}")

        # Update learning rate every 200 epoch
        base_net_20_optimizer_schedular.step()
        base_net_32_optimizer_schedular.step()
        base_net_32_adam_optimizer_schedular.step()
        res_net_20_optimizer_schedular.step()
        res_net_32_optimizer_schedular.step()
        res_net_32_adam_optimizer_schedular.step()

        # Update accuracy lists for current epoch
        base_net_20_epoch_training_accuracy.append(100 * base_net_20_epoch_training_accurate / total_train_data)
        base_net_32_epoch_training_accuracy.append(100 * base_net_32_epoch_training_accurate / total_train_data)
        base_net_32_training_accuracy_for_adam.append(100 * base_net_32_epoch_training_accurate_for_adam / total_train_data)
        res_net_20_epoch_training_accuracy.append(100 * res_net_20_epoch_training_accurate / total_train_data)
        res_net_32_epoch_training_accuracy.append(100 * res_net_32_epoch_training_accurate / total_train_data)
        res_net_32_training_accuracy_for_adam.append(100 * res_net_32_epoch_training_accurate_for_adam / total_train_data)

        # Update losses lists for current epoch
        base_net_20_epoch_training_losses.append(base_net_20_epoch_training_loss / total_train_data)
        base_net_32_epoch_training_losses.append(base_net_32_epoch_training_loss / total_train_data)
        base_net_32_training_losses_for_adam.append(base_net_32_epoch_training_loss_for_adam / total_train_data)
        res_net_20_epoch_training_losses.append(res_net_20_epoch_training_loss / total_train_data)
        res_net_32_epoch_training_losses.append(res_net_32_epoch_training_loss / total_train_data)
        res_net_32_training_losses_for_adam.append(res_net_32_epoch_training_loss / total_train_data)

        # -------------------- Training finished ---------------------

        # initialization for testing experiments
        base_net_20_epoch_testing_accurate, base_net_32_epoch_testing_accurate = 0, 0
        base_net_20_epoch_testing_loss, base_net_32_epoch_testing_loss = 0, 0
        res_net_20_epoch_testing_accurate, res_net_32_epoch_testing_accurate = 0, 0
        res_net_20_epoch_testing_loss, res_net_32_epoch_testing_loss = 0, 0
        total_test_data = 0

        # Disabling gradient calculation will reduce memory consumption
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                total_test_data += len(labels)

                # BaseNet-20 Testing
                base_net_20_outputs = base_net_20(inputs)
                _, predicted_20 = torch.max(base_net_20_outputs, 1)
                base_net_20_epoch_testing_accurate += (predicted_20 == labels).sum().item()

                base_net_20_loss = criterion(base_net_20_outputs, labels)
                base_net_20_epoch_testing_loss += base_net_20_loss.item()

                # BaseNet-32 Testing
                base_net_32_outputs = base_net_32(inputs)
                _, predicted_32 = torch.max(base_net_32_outputs, 1)
                base_net_32_epoch_testing_accurate += (predicted_32 == labels).sum().item()

                base_net_32_loss = criterion(base_net_32_outputs, labels)
                base_net_32_epoch_testing_loss += base_net_32_loss.item()

                # ResNet-20 Testing loss
                res_net_20_outputs = res_net_20(inputs)
                _, predicted_20 = torch.max(res_net_20_outputs, 1)
                res_net_20_epoch_testing_accurate += (predicted_20 == labels).sum().item()

                res_net_20_loss = criterion(res_net_20_outputs, labels)
                res_net_20_epoch_testing_loss += res_net_20_loss.item()

                # ResNet-32 Testing loss
                res_net_32_outputs = res_net_32(inputs)
                _, predicted_32 = torch.max(res_net_32_outputs, 1)
                res_net_32_epoch_testing_accurate += (predicted_32 == labels).sum().item()

                res_net_32_loss = criterion(res_net_32_outputs, labels)
                res_net_32_epoch_testing_loss += res_net_32_loss.item()

        base_net_20_epoch_testing_accuracy.append(100 * base_net_20_epoch_testing_accurate / total_test_data)
        base_net_32_epoch_testing_accuracy.append(100 * base_net_32_epoch_testing_accurate / total_test_data)
        res_net_20_epoch_testing_accuracy.append(100 * res_net_20_epoch_testing_accurate / total_test_data)
        res_net_32_epoch_testing_accuracy.append(100 * res_net_32_epoch_testing_accurate / total_test_data)

        base_net_20_epoch_testing_losses.append(base_net_20_epoch_testing_loss / total_test_data)
        base_net_32_epoch_testing_losses.append(base_net_32_epoch_testing_loss / total_test_data)
        res_net_20_epoch_testing_losses.append(res_net_20_epoch_testing_loss / total_test_data)
        res_net_32_epoch_testing_losses.append(res_net_32_epoch_testing_loss / total_test_data)

        print(f"Epoch {epoch}: generated results till now:")
        print(base_net_20_epoch_training_accuracy, base_net_32_epoch_training_accuracy)
        print(base_net_20_epoch_testing_accuracy, base_net_32_epoch_testing_accuracy)
        print(base_net_20_epoch_training_losses, base_net_32_epoch_training_losses)
        print(base_net_20_epoch_testing_losses, base_net_32_epoch_testing_losses)
        print(base_net_32_training_accuracy_for_adam, base_net_32_training_losses_for_adam)

        print(res_net_20_epoch_training_accuracy, res_net_32_epoch_training_accuracy)
        print(res_net_20_epoch_testing_accuracy, res_net_32_epoch_testing_accuracy)
        print(res_net_20_epoch_training_losses, res_net_32_epoch_training_losses)
        print(res_net_20_epoch_testing_losses, res_net_32_epoch_testing_losses)
        print(res_net_32_training_accuracy_for_adam, res_net_32_training_losses_for_adam)

    print(base_net_20_epoch_training_accuracy, base_net_32_epoch_training_accuracy)
    print(base_net_20_epoch_testing_accuracy, base_net_32_epoch_testing_accuracy)
    print(base_net_20_epoch_training_losses, base_net_32_epoch_training_losses)
    print(base_net_20_epoch_testing_losses, base_net_32_epoch_testing_losses)
    print(base_net_32_training_accuracy_for_adam, base_net_32_training_losses_for_adam)

    print(res_net_20_epoch_training_accuracy, res_net_32_epoch_training_accuracy)
    print(res_net_20_epoch_testing_accuracy, res_net_32_epoch_testing_accuracy)
    print(res_net_20_epoch_training_losses, res_net_32_epoch_training_losses)
    print(res_net_20_epoch_testing_losses, res_net_32_epoch_testing_losses)
    print(res_net_32_training_accuracy_for_adam, res_net_32_training_losses_for_adam)

    # ----------------------- Plotting experiment results ------------------------------------------------

    # --------- Compare the training and testing accuracy over epoch for BaseNet-20 and BaseNet-32 -------
    results1 = [
        base_net_20_epoch_training_accuracy, base_net_32_epoch_training_accuracy,
        base_net_20_epoch_testing_accuracy, base_net_32_epoch_testing_accuracy
    ]

    labels1 = [
        "base_net_20_training_accuracy", "base_net_32_training_accuracy",
        "base_net_20_testing_accuracy", "base_net_32_testing_accuracy"
    ]

    PlotGeneration.line_plotting(results1, labels1, y_label="Accuracy (%)",
                                 title="Accuracy between BaseNet 20 & 32")

    # -------- Compare the training and testing loss over epoch for ResNet-20 and ResNet-32 -----------
    results2 = [
        res_net_20_epoch_training_losses, res_net_32_epoch_training_losses,
        res_net_20_epoch_testing_losses, res_net_32_epoch_testing_losses
    ]

    labels2 = [
        "res_net_20_training_losses", "res_net_32_training_losses",
        "res_net_20_testing_losses", "res_net_32_testing_losses"
    ]

    PlotGeneration.line_plotting(results2, labels2, title="Loss between ResNet 20 & 32")

    # ------- Compare the training and testing accuracy over epoch for BaseNet-20 and ResNet-20 ----------
    results3 = [
        base_net_20_epoch_training_accuracy, res_net_20_epoch_training_accuracy,
        base_net_20_epoch_testing_accuracy, res_net_20_epoch_testing_accuracy
    ]

    labels3 = [
        "base_net_20_training_accuracy", "res_net_20_training_accuracy",
        "base_net_20_testing_accuracy", "res_net_20_testing_accuracy"
    ]

    PlotGeneration.line_plotting(results3, labels3, y_label="Accuracy (%)",
                                 title="Accuracy between BaseNet-20 & ResNet-20")

    # ------  Compare the training and testing accuracy over epoch for BaseNet-32 and ResNet-32 ---------
    results4 = [
        base_net_32_epoch_training_accuracy, res_net_32_epoch_training_accuracy,
        base_net_32_epoch_testing_accuracy, res_net_32_epoch_testing_accuracy
    ]

    labels4 = [
        "base_net_32_training_accuracy", "res_net_32_training_accuracy",
        "base_net_32_testing_accuracy", "res_net_32_testing_accuracy"
    ]

    PlotGeneration.line_plotting(results4, labels4, y_label="Accuracy (%)",
                                 title="Accuracy between BaseNet-32 & ResNet-32")

    # ------ Compare the training and testing loss over epoch for BaseNet-20 and ResNet-20
    results5 = [
        base_net_20_epoch_training_losses, res_net_20_epoch_training_losses,
        base_net_20_epoch_testing_losses, res_net_20_epoch_testing_losses
    ]

    labels5 = [
        "base_net_20_training_losses", "res_net_20_training_losses",
        "base_net_20_testing_losses", "res_net_20_testing_losses"
    ]

    PlotGeneration.line_plotting(results5, labels5, title="Loss between BaseNet-20 & ResNet-20")

    # ------ Compare the training and testing loss over epoch for BaseNet-32 and ResNet-32
    results6 = [
        base_net_32_epoch_training_losses, res_net_32_epoch_training_losses,
        base_net_32_epoch_testing_losses, res_net_32_epoch_testing_losses
    ]

    labels6 = [
        "base_net_32_training_loss", "res_net_32_training_loss",
        "base_net_32_testing_loss", "res_net_32_testing_loss"
    ]

    PlotGeneration.line_plotting(results6, labels6, title="Loss between BaseNet-32 & ResNet-32")

    # ------ Compare the training accuracy over epochs
    # ------ for BaseNet-32 and ResNet-32 model using different optimizer
    results7 = [
        base_net_32_training_accuracy_for_adam, res_net_32_training_accuracy_for_adam
    ]

    labels7 = [
        "base_net_32_training_accuracy", "res_net_32_training_accuracy"
    ]

    PlotGeneration.line_plotting(results7, labels7, y_label="Accuracy (%)", title="Accuracy between BaseNet-32 & ResNet-32 with different optimizer")

    # ------ Compare the training loss over epochs
    # ------ for BaseNet-32 and ResNet-32 model using different optimizer
    results8 = [
        base_net_32_training_losses_for_adam, res_net_32_training_losses_for_adam
    ]

    labels8 = [
        "base_net_32_training_losses", "res_net_32_training_losses"
    ]

    PlotGeneration.line_plotting(results8, labels8, title="Loss between BaseNet-32 & ResNet-32 using different optimizer")
