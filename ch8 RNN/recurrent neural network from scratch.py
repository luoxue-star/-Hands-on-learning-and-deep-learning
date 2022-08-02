import torch
import math
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


def get_params(vocab_size, num_hiddens, device):
    """
    得到初始化的参数
    :param vocab_size: 词元的数量（在这里是26个字母+空格+unknown，共28个）
    :param num_hiddens: 隐层的维度数
    :param device: GPU or CPU
    :return: RNN的参数
    """
    # 输入和输出都是一个独热编码，也就是一个字母
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.1

    # RNN的参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 将每个参数添加到计算图中，以便计算梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    """
    初始化RNN的隐藏状态
    :param batch_size:
    :param num_hiddens:
    :param device:
    :return:
    """
    # 由于LSTM网络有多个起始状态，所以这里使用元组的形式
    return torch.zeros((batch_size, num_hiddens), device=device),


def rnn(inputs, state, params):
    # 输入的维度（时间步数， 批量大小， 词表向量长度）
    W_xh, W_hh, b_h, W_hq, b_q = params
    # 加上逗号，使得H不是一个元组类型（具体见上面初始化状态的方法）
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # Y是下一个隐层状态
        Y = torch.mm(H, W_hq) + b_q
        # 用output保存隐层状态
        outputs.append(Y)
    # 返回的维度就是（时间步数，批量大小，隐层维度）
    return torch.cat(outputs, dim=0), (H, )


class RNNModelScratch:
    """
    RNN网络的实现
    """
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    # __call__重载了()运算符，可以直接使用实例变量()调用该函数
    def __call__(self, X, state):
        # X是（批量大小，时间步数），转置之后，对每一列进行one-hot编码。
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    在prefix后预测新的字符
    :param prefix: 要训练的一段字符串
    :param num_preds: 要预测的字符个数
    :param net: 定义的RNN网络
    :param vocab: 一个实例变量，含有将词和索引相互转化的方法
    :param device: GPU or CPU
    :return:
    """
    # 获取初始状态
    state = net.begin_state(batch_size=1, device=device)
    # 得到第一个字符的编码值
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:
        # outputs[-1]是当前最后一个输入
        # 使用最后一个输入不断更新当前的隐藏层状态，但不进行预测，这个时期称为预热期。
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        # 本质上是一个分类问题。因为y的列维度是特征，所以找到最大的地方即为预测值
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # "".join表示直接将所有字符连接成字符串
    return "".join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """
    梯度裁剪
    :param net:RNN网络
    :param theta: 梯度阈值
    :return:
    """
    # 若是nn.Module类型，需要取出要求梯度的参数
    # 若不是，直接取出自定义的网络的参数
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 这里的梯度裁剪是使用求和的方式，本质上不够严谨，最好是每个参数都计算梯度，再裁剪
    norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练一个网络
    :param net:RNN网络
    :param train_iter:迭代数据
    :param loss: 损失函数
    :param updater: 优化器
    :param device: CPU or GPU
    :param use_random_iter: 使用随机分区还是顺序分区
    :return:
    """
    state, timer = None, d2l.Timer()
    # 定义一个累加器,分别计算损失值和总的个数
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或者使用随机分区时，随机初始化隐层状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GPU是一个张量
                # 调用detach_(),表示后面的不会再向state求梯度了，就相当于断开了
                state.detach_()
            else:
                # 如果有多个就遍历每一个，例如LSTM
                for s in state:
                    s.detach_()

        # 变成一维的
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    # 返回第一个参数为困惑度
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """
    训练模型
    :param net:
    :param train_iter:
    :param vocab:
    :param lr: 学习率
    :param num_epochs:
    :param device:
    :param use_random_iter:
    :return:
    """
    loss = nn.CrossEntropyLoss()
    # 可视化
    animator = d2l.Animator(xlabel="epoch", ylabel="perplexity",
                            legend=["train"], xlim=[10, num_epochs])

    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict("time traveller"))
            animator.add(epoch+1, [ppl])

    print(f"困惑度：{ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}")
    print(predict("time traveller"))
    print(predict("traveller"))


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params, init_rnn_state, rnn)
    epochs, lr = 1000, 1
    train_ch8(net, train_iter, vocab, lr, epochs, d2l.try_gpu())



