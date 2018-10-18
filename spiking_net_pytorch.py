import utils
import numpy as np
import torch


def forward_pass(x, layers, voltages, random_tensor, counts):
    threshold = 1.0

    counts.zero_()
    for voltage in voltages:
        voltage.zero_()

    # Prepare the spiking input
    x = torch.tensor(x.astype(np.float32)).cuda()
    random_tensor.uniform_()
    spiking_input = (random_tensor < x).type(torch.cuda.FloatTensor)

    for i in range(spiking_input.shape[1]):
        x = spiking_input[:, i, :]

        for layer, voltage in zip(layers, voltages):
            voltage += torch.matmul(x, layer)
            x = voltage > threshold
            voltage[x] = 0.0
            x = x.type(torch.cuda.FloatTensor)

        counts += x

    predictions = torch.argmax(counts, dim=1).cpu().numpy()
    return predictions


def init(batch_size, seq_len, layers):
    voltages = [torch.zeros(batch_size, layer.shape[1]).cuda() for layer in layers]
    random_tensor = torch.zeros(batch_size, seq_len, 784).cuda()
    counts = torch.zeros(batch_size, 10).cuda()

    return voltages, random_tensor, counts


def main():
    layers = utils.load_weights()
    layers = [torch.tensor(layer.astype(np.float32)).cuda() for layer in layers]
    X, y = utils.load_data()

    N = X.shape[0]
    batch_size = 1000
    num_batches = N // batch_size
    seq_len = 200

    voltages, random_tensor, counts = init(batch_size, seq_len, layers)

    correct = 0
    for batch_i in range(num_batches):
        batch_X, batch_y, end = utils.get_batch(X, y, batch_size, batch_i)

        predictions = forward_pass(batch_X, layers, voltages, random_tensor, counts)
        correct += np.sum(predictions == batch_y)
        print('{}/{}  Accuracy: {:.3f}'.format(correct, end, correct/end))


if __name__ == "__main__":
    main()