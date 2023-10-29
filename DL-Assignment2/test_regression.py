from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlp import MLP, mse_loss

net = MLP(
    linear_1_in_features=2,
    linear_1_out_features=20,
    g_function1='relu',
    linear_2_in_features=20,
    linear_2_out_features=5,
    g_function2='identity'
)
x = torch.randn(10, 2)
y = torch.randn(10, 5)

# This is just one pass
net.clear_grad_and_cache()
y_hat = net.forward(x)
L, dLdy_hat = mse_loss(y, y_hat)
net.backward(dLdy_hat)

# ------------------------------------------------
# compare the result with autograd
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(2, 20)),
        ('relu', nn.ReLU()),
        ('linear2', nn.Linear(20, 5)),
    ])
)
net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_autograd = net_autograd(x)
L_autograd = F.mse_loss(y_hat_autograd, y)

net_autograd.zero_grad()
L_autograd.backward()

print((net_autograd.linear1.weight.grad.data - net.grads['dLdW1']).norm() < 1e-3)
print((net_autograd.linear1.bias.grad.data - net.grads['dLdb1']).norm() < 1e-3)
print((net_autograd.linear2.weight.grad.data - net.grads['dLdW2']).norm() < 1e-3)
print((net_autograd.linear2.bias.grad.data - net.grads['dLdb2']).norm() < 1e-3)
# ------------------------------------------------
