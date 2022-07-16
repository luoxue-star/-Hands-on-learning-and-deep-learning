import random
from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt


def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成一个小批量子序列
    :param corpus: 词汇库
    :param batch_size:
    :param num_steps: 每个数据的长度
    :return:
    """
    # 随机产生一个初始的偏移量
    corpus = corpus[random.randint(0, num_steps-1):]
    # 减去一，因为需要考虑输出，也就是标签
    num_subseqs = (len(corpus)-1) // num_steps
    # 找到每个起始位置
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，来自两个相邻的、随机的、小批量的子序列在不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回pos位置开始的num_steps的序列
        return corpus[pos:pos+num_steps]

    # 得到batch数
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 得到一个batch中的所以起始位置
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        # 生成一个迭代器
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    使用顺序分区生成一个小批量子序列（也就是相邻的batch间是相邻的）
    :param corpus: 词汇库
    :param batch_size:
    :param num_steps:
    :return:
    """
    offset = random.randint(0, num_steps)
    # 减掉1是标签问题，同上一个方法
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset:offset+num_tokens])
    Ys = torch.tensor(corpus[offset+1:offset+1+num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i:i+num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器，就是上面两个方法的封装"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


if __name__ == "__main__":
    # 得到所有的词汇
    tokens = d2l.tokenize(d2l.read_time_machine())
    # 因为每个文本行不一定是一个句子或者一个段落，因此我们将所有的文本行全部合成一个列表中
    corpus = [token for line in tokens for token in line]
    vocab = d2l.Vocab(corpus)
    # 打印词频前10的单词
    # print(vocab.token_freqs[:10])

    # 可视化词频的对数曲线（一元语法的）
    freqs = [freq for token, freq in vocab.token_freqs]
    # d2l.plot(freqs, xlabel="token:x", ylabel="frequency:n(x)", xscale="log", yscale="log")
    # plt.show()

    # 二元语法
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = d2l.Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    # 三元语法
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = d2l.Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    # 可视化一元、二元、三元语法的词频变化
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    # d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel="token:x",
    #          ylabel="frequency:n(x)", xscale="log", yscale="log",
    #          legend=["unigram", "bigram", "trigram"])
    # plt.show()




