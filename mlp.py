import utils
import numpy as np


def relu(x):
    return np.where(x > 0, x, np.zeros_like(x))


def forward_pass(layers, X, y):

    layer_activations = [relu, relu, lambda x: x]

    print('Reciprocal of largest activations per layer:')
    print('(Used for weight scaling)')
    for layer, activation in zip(layers, layer_activations):
        X = activation(X.dot(layer))
        print(1 / np.percentile(X, 99))

    y_hat = np.argmax(X, axis=1)
    accuracy = np.mean(y_hat == y)
    print('\nAccuracy: {}'.format(accuracy))


def main():
    layers = utils.load_weights()
    X, y = utils.load_data()

    forward_pass(layers, X, y)

if __name__ == "__main__":
    main()