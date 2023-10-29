from collections import OrderedDict

import torch
import torch.nn as nn
from mlp import MLP, ce_loss


net = MLP(
    linear_1_in_features=2,
    linear_1_out_features=20,
    g_function1='sigmoid',
    linear_2_in_features=20,
    linear_2_out_features=5,
    g_function2='sigmoid'
)
x = torch.randn(10, 2)
y = (torch.randn(10, 5) < 0.5) * 1.0

# This is just one pass
net.clear_grad_and_cache()
y_hat = net.forward(x)
L, dLdy_hat = ce_loss(y, y_hat)
net.backward(dLdy_hat)

#------------------------------------------------
# compare the result with autograd
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(2, 20)),
        ('sigmoid1', nn.Sigmoid()),
        ('linear2', nn.Linear(20, 5)),
        ('sigmoid2', nn.Sigmoid()),
    ])
)
net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_autograd = net_autograd(x)

L_autograd = torch.nn.BCELoss()(y_hat_autograd, y)

net_autograd.zero_grad()
L_autograd.backward()

print((net_autograd.linear1.weight.grad.data - net.grads['dLdW1']).norm() < 1e-3)
print((net_autograd.linear1.bias.grad.data - net.grads['dLdb1']).norm() < 1e-3)
print((net_autograd.linear2.weight.grad.data - net.grads['dLdW2']).norm() < 1e-3)
print((net_autograd.linear2.bias.grad.data - net.grads['dLdb2']).norm()< 1e-3)
#------------------------------------------------
