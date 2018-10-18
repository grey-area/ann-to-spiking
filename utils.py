import numpy as np


def load_weights():
    layers = [np.load('weights/layer{}.npy'.format(i)) for i in range(3)]
    scaling_factors = [0.988, 0.7708, 0.05]
    return [layer * s for layer, s in zip(layers, scaling_factors)]


def load_data():
    X = np.load('data/test_inputs.npy')
    y = np.load('data/test_labels.npy')
    return X, y


def get_batch(X, y, batch_size, batch_i):
    start = batch_size * batch_i
    end = batch_size * (batch_i + 1)
    batch_X = X[start:end, :]
    batch_X = np.expand_dims(batch_X, axis=1)
    batch_y = y[start:end]

    return batch_X, batch_y, end