import torch
from d2l import torch as d2l


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap="Reds"):
    """显示矩阵热力图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True,
                                 squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    d2l.plt.show()


if __name__ == "__main__":
    # 课后小作业
    weights = torch.rand((10, 10), dtype=torch.float64)
    exp_matrix = torch.exp(weights)
    exp_vector = torch.sum(exp_matrix, dim=1)
    attention_weights = (exp_matrix / exp_vector).reshape((1, 1, 10, 10))  # 输入是四维的
    show_heatmaps(attention_weights, xlabel="keys", ylabel="queries")






