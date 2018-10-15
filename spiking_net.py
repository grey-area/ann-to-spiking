import numpy as np
import sys

layers = [np.load('layer{}.npy'.format(i)) for i in range(1, 4, 1)]
scaling_factors = [0.37, 1.25, 0.8]
layers = [layer * s for layer, s in zip(layers, scaling_factors)]

X = np.load('test_inputs.npy')
Y = np.load('test_labels.npy')

N = 10000
seq = 200

def run(spiking_input, layers):

    threshold = 1.0
    voltages = [np.zeros(1200), np.zeros(1200), np.zeros(10)]

    counts = np.zeros(10, dtype=np.int32)

    for i in range(50):
        x = spiking_input[i, :]

        for layer_i in range(3):
            voltages[layer_i] += x.dot(layers[layer_i])
            x = voltages[layer_i] > threshold
            voltages[layer_i][x] = 0.0

        counts += x

    prediction = np.argmax(counts)
    return prediction

correct = 0

for i in range(N):
    x = X[i, :]
    y = Y[i]

    spiking_input = np.random.random((200, 784)) < np.expand_dims(x, 0)

    prediction = run(spiking_input, layers)
    correct += (prediction == y)

    print(correct / (i + 1))