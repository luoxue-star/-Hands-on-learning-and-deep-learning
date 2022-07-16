import re
from d2l import torch as d2l
import collections


def read_time_machine():
    """将时间机器数据集加载到文本行的列表中，并且将文本都转化为字母和空格"""
    # 第一次下载使用下面这一行
    # with open(d2l.download("time_machine"), 'r') as f:
    with open("../data/timemachine.txt", 'r') as f:
        # readlines表示一次读取所有内容，并且按行存储
        lines = f.readlines()
    # strip是移除字符串中的头和尾的空格或换行符号
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token="word"):
    """将文本拆分为单词或字符词元"""
    if token == "word":
        # 默认使用空格和换行符进行分割
        return [line.split() for line in lines]
    elif token == "char":
        # list就直接将一句话转化成一个个字符了
        return [list(line) for line in lines]
    else:
        print("错误：未知词元类型：" + token)


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        :param tokens:经过拆分之后的词元
        :param min_freq: 单词出现的频率的最小次数
        :param reserved_tokens: 是否反转
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 得到每个词元的频率
        counter = count_corpus(tokens)
        # 按照频率为每个词元进行排序，大的在前，小的在后
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ["<unknown>"] + reserved_tokens
        # 转化为词元：索引（本质上是按照频率次数给索引的）
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                # 若小于最小的频率直接结束
                break
            if token not in self.token_to_idx:
                # 对于没有的词进行索引
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            # 使用字典中get，若字典中没有该词，则直接为该词赋值为unk
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    # 加上@property后，直接通过.unk就可以访问了，不需要加上括号
    @property
    def unk(self):
        """未知词元的索引为0"""
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 为2维列表时，将词元列表展成一个一维列表
        tokens = [token for line in tokens for token in line]
    # 统计每个元素的个数
    return collections.Counter(tokens)


def load_corpus_time_machine(max_tokens=-1):
    """返回数据集的词元索引列表和词表"""
    # 集成了所以的类和方法
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    # 数据集本身中每个文本行不一定是一个句子或者一个段落，所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


if __name__ == "__main__":
    """
    文本预处理步骤：
    1、将文本作为字符串加载到内存中
    2、将字符串拆解（单词或者字符）
    3、建立一个此表，将拆分的次元映射到数字索引上
    4、将文本转换为数字索引
    """
    # 下载时使用
    # d2l.DATA_HUB["time_machine"] = (d2l.DATA_URL + "timemachine.txt", "090b5e7e70c295757f55df93cb0a180b9691891a")
    # lines = read_time_machine()
    # print(f"# 文本总行数：{len(lines)}")
    # print(lines[0])
    # print(lines[10])

    # tokens = tokenize(lines)
    # print(tokens[0])

    # vocab = Vocab(tokens)
    # print(list(vocab.token_to_idx.items())[:10])

    corpus, vocab = load_corpus_time_machine()
    print(len(corpus))
    print(len(vocab))

