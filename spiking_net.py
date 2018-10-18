import utils
import numpy as np


def forward_pass(batch_size, spiking_input, layers):

    threshold = 1.0
    voltages = [np.zeros((batch_size, layer.shape[1])) for layer in layers]
    counts = np.zeros((batch_size, 10), dtype=np.int32)

    for i in range(spiking_input.shape[1]):
        x = spiking_input[:, i, :]

        for layer, voltage in zip(layers, voltages):
            voltage += x.dot(layer)
            x = voltage > threshold
            voltage[x] = 0.0

        counts += x

    predictions = np.argmax(counts, axis=1)
    return predictions


def main():
    layers = utils.load_weights()
    X, y = utils.load_data()

    N = X.shape[0]
    batch_size = 100
    num_batches = N // batch_size

    seq_len = 200

    correct = 0
    for batch_i in range(num_batches):
        batch_X, batch_y, end = utils.get_batch(X, y, batch_size, batch_i)

        spiking_input = np.random.random((batch_size, seq_len, 784)) < batch_X

        predictions = forward_pass(batch_size, spiking_input, layers)
        correct += np.sum(predictions == batch_y)
        print('{}/{}  Accuracy: {:.3f}'.format(correct, end, correct/end))


if __name__ == "__main__":
    main()