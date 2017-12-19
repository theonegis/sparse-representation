import numpy as np
from matplotlib import pyplot as plt
from skimage.util import view_as_blocks, view_as_windows
# import scipy.io as sio

from image_utils import blocks_to_image
from least_square_dict_learning import RLSDictionaryLearning
from dict_learning_utils import mse, psnr, sparsity

if __name__ == '__main__':
    img_path = '/Users/tanzhenyu/Resources/DataWare/StarFM/reflectance_test/L7SR.05-24-01.green.dat'
    im_data = np.fromfile(img_path, dtype=np.int16)
    im_data = im_data.reshape(1200, 1200).astype(np.float)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im_data)

    n_components = 70
    patch_size = (100, 100)
    im_patches = view_as_windows(im_data, patch_size, step=80)
    im_patches = im_patches.reshape(-1, patch_size[0], patch_size[1])

    model = RLSDictionaryLearning(n_components, n_nonzero=0.5 * patch_size[0])
    model.fit(im_patches, n_num=im_patches.shape[0])
    dictionary = model.dictionary
    # sio.savemat('rs_dict.mat', {'dict': dictionary})

    im_blocks = view_as_blocks(im_data, patch_size)
    reconstruct_blocks = np.zeros(im_blocks.shape)
    for i in range(im_blocks.shape[0]):
        for j in range(im_blocks.shape[1]):
            code = model.transform(im_blocks[i, j])
            # name = 'x' + str(i) + 'y' + str(j)
            # sio.savemat('rs_code_' + name + '.mat', {name: code})
            print('the sparsity of the %dth patch is: %.5f' %
                  (i * im_blocks.shape[1] + (j + 1), sparsity(code)))
            reconstruct_blocks[i, j] = dictionary.dot(code)
    reconstructed = blocks_to_image(reconstruct_blocks, im_data.shape)
    print('the MSE of the reconstructed image is %.5f' % mse(im_data, reconstructed))
    print('the PSNR of the reconstructed image is %.5f' % psnr(im_data, reconstructed))

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed)
    plt.show()
