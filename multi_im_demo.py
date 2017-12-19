import numpy as np
from PIL import Image
import urllib.request
import io
from matplotlib import pyplot as plt
from skimage.util import view_as_blocks, view_as_windows

from image_utils import blocks_to_image
from least_square_dict_learning import RLSDictionaryLearning

if __name__ == '__main__':
    img_url = 'http://www.cs.cmu.edu/~chuck/lennapg/lena_std.tif'
    with urllib.request.urlopen(img_url) as url:
        img_file = io.BytesIO(url.read())
    im_lenna = Image.open(img_file).convert('L')
    im_lenna = np.array(im_lenna).astype(float)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_lenna, cmap='gray')

    n_components = 70
    patch_size = (64, 64)
    im_patches = view_as_windows(im_lenna, patch_size, step=50)
    im_patches = im_patches.reshape(-1, patch_size[0], patch_size[1])

    model = RLSDictionaryLearning(n_components=n_components)
    model.fit(im_patches, n_num=im_patches.shape[0])
    dictionary = model.dictionary

    im_blocks = view_as_blocks(im_lenna, (64, 64))
    reconstruct_blocks = np.zeros(im_blocks.shape)
    for i in range(im_blocks.shape[0]):
        for j in range(im_blocks.shape[1]):
            code = model.transform(im_blocks[i, j])
            reconstruct_blocks[i, j] = dictionary.dot(code)
    reconstructed = blocks_to_image(im_blocks, im_lenna.shape)

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.show()
