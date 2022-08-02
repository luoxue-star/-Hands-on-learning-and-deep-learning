import collections
import math

import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l


class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        """
        :param vocab_size: 单词的数量
        :param embed_size: embedding的维数
        :param num_hiddens: 隐层维度的数量
        :param num_layers: encoder中RNN的层数
        :param dropout: dropout的数值
        :param kwargs:
        """
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)  # 传入X的维度为(batch_size, num_steps, embed_size)
        X = X.permute(1, 0, 2)  # 在RNN中X的维度应将batch_size和num_steps交换
        output, state = self.rnn(X)  # 初始状态默认为0
        return output, state


class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 为了进一步包含经过解码的输入序列的信息，上下文的变量在每一个时间步上和解码器的输出拼接，所以维度是embed_size+num_hiddens
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.linear = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]  # 索引1表示是state

    def forward(self, X, state):
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)  # 使解码器的输出状态和X有相同的num_steps
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.linear(output).permute(1, 0, 2)  # output最终的输出为(batch_size, num_steps, num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项,本质上就是将有效长度之后的全部设置为0
    :param X: 二维的tensor
    :param valid_len: 一维的tensor
    :param value: 填充无效值，默认用0表示
    :return: 屏蔽无效的值后的X
    """
    max_len = X.size(1)  # 时间步数
    # 将有效长度之后置为False
    mask = torch.arange((max_len), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value  # 将有效长度后的值置为0
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮盖的交叉熵损失函数"""
    # pred:(batch_size, num_steps, vocab_size)
    # label:(batch_size,num_steps)
    # valid_len:(batch_size, )
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = "none"
        # pytorch计算损失通常将num_steps放在最后
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def xavier_init_weights(m):
        """xavier参数初始化"""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            # 获取各个参数的name
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel="epoch", ylabel="loss", xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab["<bos>"]] * Y.shape[0], device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 每个输入都加上<bos>开始标志，且每个输入都去掉最后一个
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f"loss:{metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f}"
          f"tokens/sec on {str(device)}")
    d2l.plt.show()


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device, save_attention_weights=False):
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']]  # 每个句子用空格分割并加上eos
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)  # 有效长度计算
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab["<pad>"])  # 截断句子或者补充句子
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)  # 添加batch维度
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)  # 得到初始隐藏状态，即就是encoder的输出
    dec_X = torch.unsqueeze(torch.tensor([tgt_vocab["<bos>"]], dtype=torch.long, device=device), dim=0)  # 添加batch维度
    output_seq, attention_weight_seq = [], []
    for step in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)  # 使用最高可能性的词元作为下一次的输入
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 预测到<eos>就表示预测结束
        if pred == tgt_vocab["<eos>"]:
            break
        output_seq.append(pred)
    return " ".join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def bleu(pred_seq, label_seq, k):
    """计算BLEU，机器翻译的评价指标, k表示n元语法"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1): 
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i:i+n])] += 1  # 求解n元语法的数量
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i:i+n])] > 0:
                num_matches += 1  # >0表示预测出现了，标签中也出现了
                label_subs[' '.join(pred_tokens[i:i+n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


if __name__ == "__main__":
    # 训练
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 预测
    English = ["go .", "i lost .", "he\'s calm .", "i\'m home ."]
    French = ["va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fre in zip(English, French):
        translation, attention_weight_seq = predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f"{eng} -> {translation}, bleu:{bleu(translation, fre, k=2):.3f}")





