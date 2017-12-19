import numpy as np
from numpy import linalg


def init_dict_code(sample, n_components, dict_init=None, code_init=None):
    # 使用SVD进行字典和稀疏编码的初始化
    if dict_init is not None and code_init is not None:
        dictionary = dict_init
        code = code_init
    else:
        u, s, v = linalg.svd(sample, full_matrices=False)
        dictionary = u
        code = np.dot(np.diag(s), v)
    rows, cols = dictionary.shape
    if n_components <= cols:
        dictionary = dictionary[:, :n_components]
        code = code[:n_components, :]
    else:
        dictionary = np.c_[dictionary, np.zeros((rows, n_components - cols))]
        code = np.r_[code, np.zeros((n_components - cols, code.shape[1]))]
    return dictionary, code


def lasso_cost(y, d, x, la=0.1):
    """
    使用Lasso模型的cost函数
    :param y: 样本
    :param d: 字典
    :param x: 稀疏系数
    :param la: lambda参数
    :return: cost函数值
    """
    return 0.5 * linalg.norm(y - d.dot(x)) + la * linalg.norm(x, ord=1)


def sparsity(mtx):
    """
    计算给定矩阵的稀疏度
    """
    return np.count_nonzero(mtx == 0) / np.prod(mtx.shape)


def mse(origin, reconstructed):
    """
    利用均方误差评价图像的重建质量
    """
    res = 0
    if origin.shape != reconstructed.shape:
        return None
    for x, y in np.ndindex(origin.shape):
        res += (origin[x, y] - reconstructed[x, y]) ** 2
    return res


def psnr(origin, reconstructed):
    """
    利用峰值信噪比评价图像的重建质量
    peak signal to noise rate (PSNR)
    """
    res = mse(origin, reconstructed)
    if res == 0:
        return np.inf
    l_max = np.max(origin)
    res = 10 * np.log10(l_max ** 2 / res)
    return res
