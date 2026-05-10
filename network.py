"""
network.py
"""

# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):

    def __init__(self, sizes):
        """
        sizes:
            List containing number of neurons in each layer.

        Example:
            [784, 30, 10]

        means:
            784 input neurons that input the pixel values of the image
            30 hidden neurons
            10 output neurons that output the probability of the image being each digit (0-9)

            Given as input through main.py
        """

        self.num_layers = len(sizes)
        self.sizes = sizes

        # Biases for all layers except input layer
        self.biases = [
            np.random.randn(y, 1)
            for y in sizes[1:]
        ]

        # Weight matrices
        self.weights = [
            np.random.randn(y, x)
            for x, y in zip(sizes[:-1], sizes[1:])
        ]

    def feedforward(self, a):
        """
        Return output of network for input 'a'
        """

        for b, w in zip(self.biases, self.weights):

            a = sigmoid(np.dot(w, a) + b)

        return a

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        test_data=None
    ):
        """
        Train network using mini-batch stochastic gradient descent.

        training_data:
            list of tuples (x, y)

        epochs:
            number of training passes

        mini_batch_size:
            size of mini batches

        eta:
            learning rate

        Given as input through main.py
        """

        training_data = list(training_data)

        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):

            random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            for mini_batch in mini_batches:

                self.update_mini_batch(
                    mini_batch,
                    eta
                )

            if test_data:

                print(
                    f"Epoch {j}: "
                    f"{self.evaluate(test_data)} / {n_test}"
                )

            else:

                print(f"Epoch {j} complete")

    def update_mini_batch(self, mini_batch, eta):
        """
        Update weights and biases using gradient descent
        on a single mini batch.
        """

        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        for x, y in mini_batch:

            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [
                nb + dnb
                for nb, dnb in zip(nabla_b, delta_nabla_b)
            ]

            nabla_w = [
                nw + dnw
                for nw, dnw in zip(nabla_w, delta_nabla_w)
            ]

        self.weights = [
            w - (eta / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]

        self.biases = [
            b - (eta / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def backprop(self, x, y):
        """
        Return tuple:
            (nabla_b, nabla_w)

        representing gradient for cost function.
        """

        nabla_b = [
            np.zeros(b.shape)
            for b in self.biases
        ]

        nabla_w = [
            np.zeros(w.shape)
            for w in self.weights
        ]

        # ---------- FEEDFORWARDING ----------

        activation = x

        activations = [x]

        zs = []

        for b, w in zip(self.biases, self.weights):

            z = np.dot(w, activation) + b

            zs.append(z)

            activation = sigmoid(z)

            activations.append(activation)

        # ---------- BACKPROP ----------

        delta = (
            self.cost_derivative(
                activations[-1],
                y
            )
            * sigmoid_prime(zs[-1])
        )

        nabla_b[-1] = delta

        nabla_w[-1] = np.dot(
            delta,
            activations[-2].transpose()
        )

        # Backpropagate through layers

        for l in range(2, self.num_layers):

            z = zs[-l]

            sp = sigmoid_prime(z)

            delta = np.dot(
                self.weights[-l + 1].transpose(),
                delta
            ) * sp

            nabla_b[-l] = delta

            nabla_w[-l] = np.dot(
                delta,
                activations[-l - 1].transpose()
            )

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Evaluate network performance on test data.
        """

        test_results = [
            (
                np.argmax(self.feedforward(x)),
                y
            )
            for (x, y) in test_data
        ]

        return sum(
            int(x == y)
            for (x, y) in test_results
        )

    def cost_derivative(
        self,
        output_activations,
        y
    ):
        """
        Return partial derivatives of cost function.
        """

        return (output_activations - y)


# ---------- ACTIVATION FUNCTIONS ----------

def sigmoid(z):
    """
    Sigmoid activation function.
    """

    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Derivative of sigmoid function.
    """

    return sigmoid(z) * (1 - sigmoid(z))