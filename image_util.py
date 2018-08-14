import numpy as np

class ImageDataConfig(object):
    def __init__(self, size, is_grayscale, is_binary):
        self.size = size
        self.is_grayscale = is_grayscale
        self.is_binary = is_binary

        self.shape = self.get_shape()

    def get_shape(self):
        if self.is_grayscale:
            return (1,) + (self.size, self.size)
        else:
            return (3,) + (self.size, self.size)

def convert_to_rgb(img, is_binary = False):
    """Given an image, make sure it has 3 channels and that it is between 0 and 1."""
    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))

    img_ch, _, _ = img.shape
    if img_ch != 3 and img_ch != 1:
        raise Exception("""Unsupported number of channels. """
                        """Must be 1 or 3, given {0}.""".format(img_ch))

    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)

    if not is_binary:
        imgp = imgp * 127.5 + 127.5
        imgp /= 255.

    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)


def compose_imgs(a, b, is_a_binary = True, is_b_binary = False):
    """Place a and b side by side to be plotted."""
    ap = convert_to_rgb(a, is_binary=is_a_binary)
    bp = convert_to_rgb(b, is_binary=is_b_binary)

    if ap.shape != bp.shape:
        raise Exception("""A and B must have the same size. """
                        """{0} != {1}""".format(ap.shape, bp.shape))

    # ap.shape and bp.shape must have the same size here
    h, w, ch = ap.shape
    composed = np.zeros((h, 2*w, ch))
    composed[:, :w, :] = ap
    composed[:, w:, :] = bp

    return composed