import time
import numpy as np
from numpy import linalg
from sklearn.utils import check_array
from sklearn.decomposition import sparse_encode
from sklearn.preprocessing import normalize

from dict_learning_utils import init_dict_code


def _exp_forget_factor(n_iter, init_value=0.99):
    a = (init_value + 1) / 2
    return 1 - (1 - init_value) * 0.5 ** (n_iter / a)


def _update_dict(y, d, x, n_iter, c):
    r = y - np.dot(d, x)
    c /= _exp_forget_factor(n_iter)
    u = np.dot(c, x)
    a = linalg.pinv(np.eye(x.shape[1], x.shape[1]) + np.dot(x.T, u))
    d += np.dot(np.dot(a, r), u.T)
    c -= np.dot(np.dot(u, a), u.T)
    return normalize(d, axis=0)


def dict_learning(samples, n_num, n_components, batch_max_iter, batch_tol,
                  n_nonzero=None, dict_init=None, inner_stat=None):
    sample = samples[0]
    print('sample shape: ' + str(sample.shape))
    if n_components is None:
        n_components = sample.shape[0]

    if dict_init is None:
        dictionary, code = init_dict_code(sample, n_components, dict_init)
    else:
        dictionary = dict_init

    t0 = time.time()

    if inner_stat is None:
        code = sparse_encode(sample.T, dictionary.T,
                             algorithm='omp', n_nonzero_coefs=n_nonzero).T
        inner_stat = linalg.pinv(code.dot(code.T))

    for i in range(n_num):
        sample = samples[i]
        check_array(sample)
        print('----------------------------------------')
        print('%dth sample involved...' % (i + 1))
        print('----------------------------------------')
        for j in range(batch_max_iter):
            # 更新稀疏编码
            code = sparse_encode(sample.T, dictionary.T,
                                 algorithm='omp', n_nonzero_coefs=n_nonzero).T
            mse = linalg.norm(sample - dictionary.dot(code))
            dt = (time.time() - t0)
            print("Iteration % 3d error: %.4f (elapsed time: %ds)" % (j + 1, mse, dt))

            if mse < batch_tol:
                break
            # 更新字典矩阵
            dictionary = _update_dict(sample, dictionary, code, j + 1, inner_stat)

    return dictionary


class RLSDictionaryLearning(object):
    def __init__(self, n_components,
                 batch_max_iter=200, batch_tol=1e-3,
                 n_nonzero=None, dict_init=None):
        self.dictionary = dict_init
        self.n_components = n_components
        self.batch_max_iter = batch_max_iter
        self.batch_tol = batch_tol
        self.n_nonzero = n_nonzero
        self.inner_stat = None

    def fit(self, samples, n_num=1):
        if samples is None:
            return None
        if n_num == 1 and samples.ndim <= 2:
            samples = np.array([samples])
        print('number of sample: %d' % n_num)
        self.dictionary = dict_learning(samples, n_num, n_components=self.n_components,
                                        batch_max_iter=self.batch_max_iter,
                                        batch_tol=self.batch_tol,
                                        n_nonzero=self.n_nonzero,
                                        inner_stat=self.inner_stat)
        return self

    def transform(self, sample):
        return sparse_encode(sample.T, self.dictionary.T,
                             algorithm='omp', n_nonzero_coefs=self.n_nonzero).T

