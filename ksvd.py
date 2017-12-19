import time
import numpy as np
from sklearn import linear_model
import scipy.misc
from matplotlib import pyplot as plt


class KSVD(object):
    def __init__(self, n_components, max_iter=60, tol=1e-6,
                 n_nonzero=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero: 稀疏度
        """
        self.dictionary = None
        self.code = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero = n_nonzero

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        rows, cols = u.shape
        if self.n_components <= cols:
            self.dictionary = u[:, :self.n_components]
        else:
            self.dictionary = np.c_[u, np.zeros((rows, self.n_components - cols))]

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, sample):
        """
        KSVD迭代过程
        """
        self._initialize(sample)
        t0 = time.time()
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, sample,
                                           n_nonzero_coefs=self.n_nonzero)
            e = np.linalg.norm(sample - np.dot(self.dictionary, x))

            dt = (time.time() - t0)
            print('Iteration % 3d error: %.4f (elapsed time: %ds)' % (i + 1, e, dt))

            if e < self.tol:
                break
            self._update_dict(sample, self.dictionary, x)

        self.code = linear_model.orthogonal_mp(self.dictionary, sample,
                                               n_nonzero_coefs=self.n_nonzero)
        return self.dictionary, self.code


if __name__ == '__main__':
    im_ascent = scipy.misc.ascent().astype(np.float)
    model = KSVD(600)
    dictionary, code = model.fit(im_ascent)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_ascent)
    im_reconstruct = dictionary.dot(code)
    plt.subplot(1, 2, 2)
    plt.imshow(im_reconstruct)
    plt.show()
