import time
import numpy as np
from numpy import linalg
from sklearn.utils import check_array
from sklearn.decomposition import sparse_encode

from dict_learning_utils import init_dict_code, lasso_cost


def _update_dict(d, stats, max_iter=300, tol=1e-4):
    """
    对于论文中的Algorithm 2：Dictionary Update过程的实现
    :param d: 字典矩阵
    :param stats: a和b状态变量
    :param max_iter: 递归过程的最大迭代次数
    :param tol: 递归过程的最大容差
    :return: 更新以后的字典
    """
    a = stats[0]
    b = stats[1]
    dd = d
    for i in range(max_iter):
        for j in range(d.shape[1]):
            if a[j, j] != 0:
                u = (b[:, j] - np.dot(d, a[:, j])) / a[j, j] + d[:, j]
                d[:, j] = u / max(np.linalg.norm(u), 1)
        if linalg.norm(d - dd) < tol:
            break
        else:
            dd = d
    return d


def dict_learning(samples, n_num, n_components, batch_max_iter, batch_tol,
                  alpha=1, dict_init=None, inner_stats=None):
    """
    使用参考文献中的Algorithm 1：Online dictionary learning算法进行在线字典求解
    :param samples: 样本，可以是单个，也可以是多个
    :param n_num: 样本数目
    :param n_components: 字典中原子的个数
    :param batch_max_iter: 单个样本字典更新过程中的最大迭代次数
    :param batch_tol: 单个样本字典更新过程中的最大容差
    :param alpha: 使用Lasso模型计算稀疏系数过程中的模型参数
    :param dict_init: 初始字典
    :param inner_stats: 初始状态变量a和b
    :return:
    """
    # 初始化参数
    sample = samples[0]
    print('sample shape: ' + str(sample.shape))
    if n_components is None:
        n_components = sample.shape[0]

    if dict_init is None:
        dictionary, code = init_dict_code(sample, n_components, dict_init)
    else:
        dictionary = dict_init
        code = np.random.rand((dictionary.shape[1], sample.shape[1]))

    if inner_stats is None:
        a = np.zeros((code.shape[0], code.shape[0]))
        b = np.zeros((sample.shape[0], code.shape[0]))
        inner_stats = [a, b]

    t0 = time.time()
    for i in range(n_num):
        sample = samples[i]
        check_array(sample)
        print('----------------------------------------')
        print('%dth sample involved...' % (i + 1))
        print('----------------------------------------')

        for j in range(batch_max_iter):
            code = sparse_encode(sample.T, dictionary.T, alpha=alpha).T
            cost_x = lasso_cost(sample, dictionary, code)
            inner_stats[0] += np.dot(code, code.T)
            inner_stats[1] += np.dot(sample, code.T)
            dictionary = _update_dict(dictionary, inner_stats)
            cost_y = lasso_cost(sample, dictionary, code)

            dt = (time.time() - t0)
            print('Iteration % 3d cost: %.4f (elapsed time: %ds)' % (j + 1, cost_y, dt))

            if abs(cost_y - cost_x) < batch_tol:
                break
    return dictionary


class OnlineDictionaryLearning(object):
    """
    参考文献：Julien Mairal, Online dictionary learning for sparse coding.
    """
    def __init__(self, n_components, batch_max_iter=200, batch_tol=1e-3,
                 alpha=1, dict_init=None):
        self.dictionary = dict_init
        self.n_components = n_components  # 字典中原子的个数
        self.batch_max_iter = batch_max_iter  # 对于每一个样本训练的最大迭代次数
        self.batch_tol = batch_tol  # 对于每一个样本迭代的最大容差
        self.alpha = alpha  # 使用Lasso模型进行稀疏编码中的参数
        self.inner_stats = None

    def fit(self, samples, n_num=1):
        """
        :param samples: 样本，可以是多个样本，也可以是单个样本
        :param n_num: 样本数目
        :return: 返回ODL模型本身
        """
        if samples is None:
            return None
        if n_num == 1 and samples.ndim <= 2:
            samples = np.array([samples])
        print('number of sample: %d' % n_num)
        self.dictionary = dict_learning(samples, n_num, n_components=self.n_components,
                                        batch_max_iter=self.batch_max_iter,
                                        batch_tol=self.batch_tol,
                                        alpha=self.alpha, inner_stats=self.inner_stats)
        return self

    def transform(self, sample):
        """
        经过fit函数得到字典以后，给定样本，得到稀疏系数
        这里调用sklearn库中的sparse_encode函数，对参数都进行转置的原因是：
        sklearn库中的模型和一般论文中的模型相反
        sklearn库中是稀疏系数乘以字典矩阵，一般论文中的模式都是字典矩阵乘以稀疏稀疏）
        :param sample: 单个样本
        :return: 稀疏系数
        """
        return sparse_encode(sample.T, self.dictionary.T, alpha=self.alpha).T
