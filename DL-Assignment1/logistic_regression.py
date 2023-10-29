import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate=0.01, max_iteration=1000, batch_size=1):
        self.learning_rate = learning_rate
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def linear_prediction(self, features):
        # Y = X.W + b
        return np.dot(features, self.weights) + self.bias

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def gradient_descent(features, labels, predictions):
        # Gradient descent of the weight and bias, dw and db respectively
        dw = (1 / len(features)) * np.dot(features.T, (predictions - labels))
        db = (1 / len(features)) * np.sum(predictions - labels)

        return dw, db

    @staticmethod
    def cross_entropy_loss(predictions, labels):
        return -np.average(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    @staticmethod
    def data_iteration(features, labels, batch_size):
        num_instances = len(features)
        indices = list(range(num_instances))

        # Make batch instances in random order
        indices = np.random.permutation(indices)

        for i in range(0, num_instances, batch_size):
            # Select "batch_size" number of instances from randomly ordered indices
            batch_indices = np.array(indices[i:min(i + batch_size, num_instances)])

            # Return selected randomly ordered batch for each coefficient update
            yield features[batch_indices], labels[batch_indices]

    def fit(self, features, labels, test_features=None, test_labels=None):
        features = np.array(features)
        self.weights = np.zeros(features.shape[1])
        self.bias = 0
        training_loss_itr = list()
        testing_loss_itr = list()

        for itr in range(self.max_iteration):
            for current_features, current_labels in self.data_iteration(features, labels, self.batch_size):
                # Apply sigmoid function on linear prediction to get logistic prediction
                predictions = self.sigmoid(self.linear_prediction(current_features))

                # Calculate gradient descent for both weight and bias parameters
                dw, db = self.gradient_descent(current_features, current_labels, predictions)

                # the minibatch stochastic gradient descent update of coefficients
                self.weights = self.weights - self.learning_rate * dw
                self.bias = self.bias - self.learning_rate * db

            # Calculate training loss per iteration
            predictions = self.sigmoid(self.linear_prediction(features))
            training_loss = self.cross_entropy_loss(predictions, labels)
            training_loss_itr.append(training_loss)

            # If test data is provided, then the testing loss is calculated per iteration
            if test_features is not None and test_labels is not None:
                predictions = self.sigmoid(self.linear_prediction(test_features))
                testing_loss = self.cross_entropy_loss(predictions, test_labels)
                testing_loss_itr.append(testing_loss)

        # Return both training and testing loss if testing data is provided, otherwise only training loss
        return (training_loss_itr, testing_loss_itr) if (test_features is not None and test_labels is not None) \
            else training_loss_itr

    def test(self, features):
        predictions = self.sigmoid(self.linear_prediction(features))

        classified_predictions = [1 if y > 0.5 else 0 for y in predictions]

        return classified_predictions
