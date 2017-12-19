import numpy as np
from PIL import Image
import urllib.request
import io
from matplotlib import pyplot as plt

from least_square_dict_learning import RLSDictionaryLearning

if __name__ == '__main__':
    img_url = 'http://www.cs.cmu.edu/~chuck/lennapg/lena_std.tif'
    with urllib.request.urlopen(img_url) as url:
        img_file = io.BytesIO(url.read())
    im_lenna = Image.open(img_file).convert('L')
    im_lenna = np.array(im_lenna).astype(float)

    n_components = 600
    model = RLSDictionaryLearning(n_components=n_components)
    model.fit(im_lenna)
    dictionary = model.dictionary
    code = model.transform(im_lenna)
    reconstructed = np.dot(dictionary, code)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_lenna, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed, cmap='gray')
    plt.show()
