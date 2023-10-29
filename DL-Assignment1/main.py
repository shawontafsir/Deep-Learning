import numpy as np

from src.data_generation import DataGeneration
from src.logistic_regression import LogisticRegression
from src.data_plotting import PlotGeneration


# Generating data
examples_count_per_class = 4096
X_train, y_train = DataGeneration.generate_data(examples_count_per_class)
X_test, y_test = DataGeneration.generate_data(examples_count_per_class)


# Q1: How to initialize the weights and bias?
batch_size = 32
clf_1 = LogisticRegression(learning_rate=0.1, max_iteration=100, batch_size=batch_size)
training_loss_itr, testing_loss_itr = clf_1.fit(X_train, y_train, X_test, y_test)

PlotGeneration.line_plotting([training_loss_itr, testing_loss_itr], labels=['train', 'test'])

y_pred = clf_1.test(X_test)


def accuracy(pred, test):
    return np.sum(pred == test) / len(test)


print(accuracy(y_pred, y_test))


# Q2: Compare stochastic, batch and mini-batch gradient descent.
clf_2 = LogisticRegression(learning_rate=0.1, max_iteration=100)
batches = [2, 32, 256, 512, 2000]
values = []
labels = []
for batch_size in batches:
    clf_2.batch_size = batch_size
    loss_itr = clf_2.fit(X_train, y_train)
    values.append(loss_itr)
    labels.append(f'Batch of {batch_size}')

PlotGeneration.line_plotting(values, labels)


# Q3: How learning rate affect the time to converge?
clf_3 = LogisticRegression(max_iteration=100, batch_size=32)
learning_rates = [0.05, 0.1, 0.2, 0.3]
values = []
labels = []
for learning_rate in learning_rates:
    clf_3.learning_rate = learning_rate
    loss_itr = clf_3.fit(X_train, y_train)
    values.append(loss_itr)
    labels.append(f'Learning rate {learning_rate}')

PlotGeneration.line_plotting(values, labels)
