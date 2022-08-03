import torch
from torch import nn
from d2l import torch as d2l


class AttentionDecoder(d2l.Decoder):
    """含注意力机制的解码器基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property  # 用于保护类属性，可以直接类内变量调用，不用当做方法调用（即就是不用加括号）
    def attention_weights(self):
        raise NotImplementedError


class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0.0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 因为此时query解码器的一个隐状态，key、value都是编码器的隐状态，
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 因为此时每个输入都是输入和隐状态的拼接
        self.rnn = nn.GRU(
            embed_size  + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.linear = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs:(batch_size,num_steps,num_hiddens)
        # hidden_state:(num_layers,batch_size,num_hiddens)
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs:(batch_size,num_steps,num_hiddens)
        # hidden_state:(num_layers,batch_size,num_hiddens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # X:(num_steps,batch_size,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # query:(batch_size,1,num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            context = self.attention(query, enc_outputs, enc_outputs, enc_valid_lens)
            # 在特征维度上拼接，x:(batch_size,1,num_hiddens+embed_size)
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # rnn输入要求是batch_size在第二个维度
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 全连接层变换后，outputs:(num_steps,batch_size,vocab_size)
        outputs = self.linear(torch.cat(outputs, dim=0))
        return (outputs.permute(1, 0, 2), [enc_outputs, hidden_state, enc_valid_lens])

    @property
    def attention_weights(self):
        return self._attention_weights


if __name__ == "__main__":
    embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, epochs, device = 0.005, 250, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encoder = d2l.Seq2SeqEncoder(
        len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqAttentionDecoder(
        len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    d2l.train_seq2seq(net, train_iter, lr, epochs, tgt_vocab, device)
    d2l.plt.show()

    English = ["go .", "i lost .", "he\'s calm .", "i\'m home ."]
    French = ["va !", "j\'ai perdu .", "il est calme .", "je suis chez moi ."]
    for eng, fre in zip(English, French):
        translation, attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device)
        print(f"{eng} -> {translation}, bleu:{d2l.bleu(translation, fre, k=2):.3f}")

