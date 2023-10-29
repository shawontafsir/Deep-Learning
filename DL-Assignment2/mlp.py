import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP:
    def __init__(
            self,
            linear_1_in_features,
            linear_1_out_features,
            g_function1,
            linear_2_in_features,
            linear_2_out_features,
            g_function2
    ):
        """
        Args:
            linear_1_in_features: the number of in features of first linear layer
            linear_1_out_features: the number of out features of first linear layer
            linear_2_in_features: the number of in features of second linear layer
            linear_2_out_features: the number of out features of second linear layer
            g_function1: string for the activation function at hidden layer: relu | sigmoid | identity
            g_function2: string for the activation function at output layer: relu | sigmoid | identity
        """
        self.g_functions = [g_function1, g_function2]

        self.parameters = dict(
            W1=torch.randn(linear_1_out_features, linear_1_in_features),
            b1=torch.randn(linear_1_out_features),
            W2=torch.randn(linear_2_out_features, linear_2_in_features),
            b2=torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dLdW1=torch.zeros(linear_1_out_features, linear_1_in_features),
            dLdb1=torch.zeros(linear_1_out_features),
            dLdW2=torch.zeros(linear_2_out_features, linear_2_in_features),
            dLdb2=torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

        # mapping activation functions with user definition
        self.g_act_function = dict(
            relu=F.relu,
            sigmoid=F.sigmoid,
            identity=nn.Identity(),
            linear=F.linear
        )

        # mapping derivatives of activation functions with user definition
        self.g_act_function_derivative = dict(
            relu=self.relu_derivative,
            sigmoid=lambda z: F.sigmoid(z) * (1 - F.sigmoid(z)),
            identity=lambda z: 1,
            linear=lambda z: 1
        )

    def relu_derivative(self, z):
        """
        :param z: tensor of any shape
        :return: derivative of relu(z)
        """
        z[z > 0] = 1
        z[z <= 0] = 0

        return z

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        self.cache["a"] = [x]
        self.cache["z"] = []

        # Iterate through hidden to output layer
        for i, g_function in enumerate(self.g_functions):
            # Get previous activation signal
            a_prev = self.cache["a"][-1]
            layer = i + 1

            # calculate current accumulated signal
            z_cur = torch.matmul(a_prev, self.parameters[f"W{layer}"].T) + self.parameters[f"b{layer}"].T

            # calculate current activation signal
            a_cur = self.g_act_function[g_function](z_cur)

            # store the current accumulated and activation signal
            self.cache["z"].append(z_cur)
            self.cache["a"].append(a_cur)

        # return the activation signal of last layer
        return self.cache["a"][-1]

    def backward(self, dLdy_hat):
        """
        Args:
            dLdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # The error at the output layer
        del_next = dLdy_hat

        # Iterate in a backward manner from the last to first layer to calculate gradients in each layer
        for i in range(len(self.cache["z"])-1, -1, -1):
            z_cur = self.cache["z"][i]   # accumulated signal
            g_function = self.g_functions[i]   # current layers activation function
            a_cur = self.cache["a"][i]   # activation signal of current layer

            # The error for current layer: del = del_next * g'(z_cur)
            del_cur = del_next * self.g_act_function_derivative[g_function](z_cur)

            # Calculate the loss w.r.t the weight of the current layer: dldw = del * a_cur
            self.grads[f"dLdW{i+1}"] = 1 / len(a_cur) * torch.matmul(del_cur.T, a_cur)

            # Calculate the loss w.r.t the bias of the current layer
            self.grads[f"dLdb{i+1}"] = 1 / len(del_cur) * torch.sum(del_cur.T, dim=1)

            # Update the error to be used in the next iteration
            del_next = torch.matmul(del_cur, self.parameters[f"W{i+1}"])

    def update_weight(self, learning_rate):
        """
        Update the weight and bias parameters based on gradient
        Iterate through number of parameters initialized in self.parameters
        Since we have 2 params, we need to update params at total_params / 2 layer
        """
        for i in range(len(self.parameters.keys())//2):
            # Update weight and bias at (i+1)th layer
            self.parameters[f"W{i + 1}"] -= learning_rate * self.grads[f"dLdW{i + 1}"]
            self.parameters[f"b{i + 1}"] -= learning_rate * self.grads[f"dLdb{i + 1}"]

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()


def mse_loss(y, y_hat):
    """
    Args:
        y: the actual tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        loss: scalar of loss
        dLdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # Calculate MSE loss
    loss = 1/2 * torch.mean(torch.pow((y - y_hat), 2))

    # Calculate derivative of loss w.r.t output
    dLdy_hat = (y_hat - y)

    return loss, dLdy_hat


def ce_loss(y, y_hat):
    """
    Args:
        y_hat: the predicted label tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dLdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # Calculate cross-entropy loss
    loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

    # Calculate derivative w.r.t output
    dLdy_hat = -(y / y_hat - (1 - y) / (1 - y_hat))

    return loss, dLdy_hat
