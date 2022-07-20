"""此代码仅仅是双向循环神经网络的使用方式，但是此代码是一个错误的应用。因为双向的循环神经网络不适用于预测未来的序列，只适用于填补空缺位置。"""
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn

if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    device = d2l.try_gpu()
    train_iter, vocab = d2l.load_data_time_machine(batch_size=batch_size, num_steps=num_steps)
    vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
    num_inputs = vocab_size
    lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
    model = d2l.RNNModel(lstm_layer, vocab_size)
    model = model.to(device)
    epochs, lr = 500, 1
    d2l.train_ch8(model, train_iter, vocab, lr, epochs, device)
    plt.show()

