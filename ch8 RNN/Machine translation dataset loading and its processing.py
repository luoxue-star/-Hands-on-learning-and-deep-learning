import torch
from d2l import torch as d2l
import os


def read_data_nmt():
    """加载英语-法语数据集"""
    data_dir = d2l.download_extract("fra-eng")
    with open(os.path.join(data_dir, "fra.txt"), 'r', encoding="utf-8") as f:
        # f.read()表示读取所有的文件，并且存在为一个大字符串
        return f.read()


def preprocess_nmt(text):
    """ 预处理 英语-法语 数据 """
    def no_space(char, prev_char):
        """若此时的字符是标点符号且前一个字符不是空格就返回Ture，否则返回False"""
        return char in set(",.!?") and prev_char != ' '

    # 使用空格代替不间断空格，使用小写字母代替大写字母
    # 去掉乱码，\xa0是拉丁扩展字符集里的字符，代表的是不间断空白符，超出了GBK编码范围，要去掉
    # \u202f是保证文件从左到右显示强制字符，用空格直接代替掉
    text = text.replace("\u202f", ' ').replace("\xa0", ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char for i, char in enumerate(text)]
    # ''代表分割符，join表示将out转为字符串，并且以''里面的元素作为分割符号
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化（单词） 英语-法语 数据集"""
    # source和target分别存储英语和法语的词元
    source, target = [], []
    # 将文本用换行符号\n表示
    for i, line in enumerate(text.split('\n')):
        # num_examples是句子个数
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')  # 因为每一行是由英语-法语组成，所以用制表符分割
        if len(parts) == 2:
            source.append(parts[0].split(' '))  # 将语句转为一个一个的单词
            target.append(parts[1].split(' '))

    # 返回的是两个列表，每个列表中存储着词元
    return source, target


def show_list_len_pair_hist(legend, x_label, y_label, x_list, y_list):
    """绘制列表长度对的直方图,也就是每个句子的长度"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in x_list], [len(l) for l in y_list]])
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
    d2l.plt.show()


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列，大于num_steps截断，小于填充"""
    # 因为每个句子的长度可能都不一样，所以小于指定长度的就填充pad，大于的直接砍掉
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))


def build_array_nmt(lines, vocab, num_steps):
    """
        将机器翻译的文本序列转换为小批量
    :param lines: 单词列表
    :param vocab: 词表
    :param num_steps: 句子长度
    :return:
    """
    lines = [vocab[l] for l in lines]  # 将单词转化为数字
    lines = [l + [vocab["<eos>"]] for l in lines]  # 为每一句话加上结尾符号
    array = torch.tensor([truncate_pad(l, num_steps, vocab["<pad>"]) for l in lines])
    valid_len = (array != vocab["<pad>"]).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    # min_freq表示单词频率若是小于2，就直接去掉
    src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    tgt_vocab = d2l.Vocab(target, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)  # valid_len是有效的长度(去掉pad、bos和eos)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


if __name__ == "__main__":
    # d2l.DATA_HUB["fra-eng"] = (d2l.DATA_URL + "fra-eng.zip", "946466ad1522d915e7b0f9296181140edcf86a4f5")
    #
    # raw_text = read_data_nmt()
    #
    # text = preprocess_nmt(raw_text)
    #
    # source, target = tokenize_nmt(text)
    #
    # show_list_len_pair_hist(["source", "target"], "# tokens per sequence", "count", source, target)
    #
    # # bos表示开始，eos表示句子结束，pad是填充内容
    # src_vocab = d2l.Vocab(source, min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>"])

    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print("X:", X.type(torch.int32))
        print("X的有效长度：", X_valid_len)
        print("Y:", Y.type(torch.int32))
        print("Y的有效长度：", Y_valid_len)














