import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 若RNN是双向的，num_directions为2，否则为1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens*2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        # Y是所有的隐层状态，而state是Y中的最后一个元素
        Y, state = self.rnn(X, state)
        # 需要先将Y转化为(时间步数*批量大小，隐层特征数)
        # 最终输出大小为(时间步数*批量大小，词表大小)
        # output是X移动一步的输出
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size,
                                self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size,
                                self.num_hiddens), device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size,
                                 self.num_hiddens), device=device))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # 初始化隐层状态为（隐藏层层数， 批量数， 隐层特征维度）
    state = torch.zeros((1, batch_size, num_hiddens))
    device = d2l.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    num_epochs, lr = 800, 1
    d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)



