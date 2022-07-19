import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size=batch_size, num_steps=num_steps)
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    device = d2l.try_gpu()
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    model = model.to(device)
    epochs, lr = 500, 1
    d2l.train_ch8(model, train_iter, vocab, lr, epochs, device)
    plt.show()



