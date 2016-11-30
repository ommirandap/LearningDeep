"""
2 Layer Neural Network implemented from scratch
No framework used! Only numpy <3

It is a classification NN to resolve the XOR problem (non-linear)
It uses the (momentum based) gradient descent algorithm
- tanh activation for hidden layer
- sigmoid for output layer
- cross-entropy loss

Created with the help of Siraj <3

"""
import numpy as np
import time

__author__ = 'ommirandap'

np.random.seed(0)

# Sigmoid activation function for the output layer


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Derivative of tanh function


def tanh_prime(x):
    return 1.0 - np.tanh(x)**2


def train(x, y_, w_1, w_2, b_1, b_2):

    # A = preactivation value for the 1st layer neurons
    # Z = activation value for the 1st layer neurons (output value)
    A = np.dot(x, w_1) + b_1
    Z = np.tanh(A)

    # B = preactivation value for the 1st layer neurons, using as input
    # the 1st layer output (Z)
    # Y = activation value for the 2nd layer neurons (output value)
    B = np.dot(Z, w_2) + b_2
    Y = sigmoid(B)

    Ew = Y - y_
    Ev = tanh_prime(A) * np.dot(w_2, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    loss = -np.mean(y_ * np.log(Y) + (1 - y_) * np.log(1 - Y))

    return loss, (dV, dW, Ev, Ew)


def predict(x, w_1, w_2, b_1, b_2):
    A = np.dot(x, w_1) + b_1
    B = np.dot(np.tanh(A), w_2) + b_2

    return (sigmoid(B) > 0.5).astype(int)


def main():
    # Variables
    n_hidden = 10
    n_in = 10
    n_out = 10
    n_sample = 1000
    learning_rate = 0.001
    momentum = 0.9
    n_epochs = 1

    w_1 = np.random.normal(scale=0.1, size=(n_in, n_hidden))
    w_2 = np.random.normal(scale=0.1, size=(n_hidden, n_out))
    b_1 = np.zeros(n_hidden)
    b_2 = np.zeros(n_out)

    params = [w_1, w_2, b_1, b_2]

    X = np.random.binomial(1, 0.5, (n_sample, n_in))
    Y = X ^ 1

    # Training time!
    for epoch in range(n_epochs):
        err = []
        upd = [0] * len(params)

        t0 = time.clock()

        for i in range(X.shape[0]):
            loss, grad = train(X[i], Y[i], *params)

            for j in range(len(params)):
                params[j] -= upd[j]

            for j in range(len(params)):
                upd[j] = learning_rate * grad[j] + momentum * upd[j]

            err.append(loss)

        print('Epoch: %d, Loss: %.8f, Time: %.4fs' %
              (epoch, np.mean(err), time.clock() - t0))

    x = np.random.binomial(1, 0.5, n_in)
    print("XOR prediction")
    print(x)
    print(predict(x, *params))


if __name__ == '__main__':
    main()
