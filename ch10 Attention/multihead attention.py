import torch
from d2l import torch as d2l
from torch import nn


def transpose_qkv(X, num_heads):
    """为使多头注意力可以进行并行运行，而变换维度"""
    # X:(batch_size,查询或键值对个数,num_hiddens)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # X:(batch_size,num_heads,查询或键值对个数,num_hiddens/num_heads)
    X = X.permute(0, 2, 1, 3)

    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """transpose_qkv的逆变换"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads,
                 dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries,keys,values:(batch_size,查询或键值对个数,num_hiddens)
        # valid_lens:(batch_size,)or(batch_size,查询个数)
        # 通过transpose_qkv后queries,keys,values变为
        # (batch_size*num_heads,查询或键值对个数,num_hiddens/num_heads)
        # 所以这也要求num_hiddens/num_heads能够整除
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在第一个维度上复制num_heads次，因为这里是多头的注意力
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)

        # output:(batch_size*num_heads,查询个数,num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)
        # output_concat:(batch_size,查询个数,num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
