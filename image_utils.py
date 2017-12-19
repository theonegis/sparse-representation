

def blocks_to_image(blocks, image_size):
    return blocks.transpose(0, 2, 1, 3).reshape(image_size)