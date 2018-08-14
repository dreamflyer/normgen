import numpy
import os
import random
import threading

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

random.seed(5000)

class SimpleFilesIterator(object):
    def __init__(self, directory, batch_size, input_config, output_config):
        self.a_dir = os.path.join(directory, 'A')
        self.b_dir = os.path.join(directory, 'B')

        a_files = set(x for x in os.listdir(self.a_dir) if x != ".DS_Store")
        b_files = set(x for x in os.listdir(self.b_dir) if x != ".DS_Store")

        self.input_config = input_config
        self.output_config = output_config

        self.filenames = list(a_files.intersection(b_files))

        random.shuffle(self.filenames)

        #self.filenames.sort(reverse = True)

        self.index = 0

        self.batch_size = batch_size

        self.filenames_generator = self.get_filenames_generator()

        self.lock = threading.Lock()

    def get_filenames_generator(self):
        filenames_length = len(self.filenames)

        while True:
            newIndex = self.index + self.batch_size

            if newIndex > len(self.filenames):
                newIndex = newIndex - filenames_length

                result = self.filenames[self.index : ] + self.filenames[ : newIndex]
            else:
                result = self.filenames[self.index : newIndex]

            self.index = newIndex

            yield result

    def normalize(self, batch, config):
        if config.is_binary:
            bin_batch = batch / 255.

            bin_batch[bin_batch >= 0.5] = 1
            bin_batch[bin_batch < 0.5] = 0

            return bin_batch
        else:
            tanh_batch = batch - 127.5
            tanh_batch /= 127.5

            return tanh_batch

    def next(self):
        with self.lock:
            file_names = next(self.filenames_generator)

        input_batch = numpy.zeros((self.batch_size,) + self.input_config.shape)
        output_batch = numpy.zeros((self.batch_size,) + self.output_config.shape)

        i = 0

        for file_name in enumerate(file_names):
            input_batch[i] = img_to_array(load_img(os.path.join(self.a_dir, file_name[1]), grayscale = self.input_config.is_grayscale, target_size = (self.input_config.size, self.input_config.size)))
            output_batch[i] = img_to_array(load_img(os.path.join(self.b_dir, file_name[1]), grayscale = self.output_config.is_grayscale, target_size = (self.output_config.size, self.output_config.size)))

            i = i + 1

        #print("files loaded: " + str(file_names))

        input_batch = self.normalize(input_batch, self.input_config)
        output_batch = self.normalize(output_batch, self.output_config)

        return [input_batch, output_batch]

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


def discriminator_generator(iterator, unet, discriminator):
    discriminator_out_size = discriminator.output_shape[-2:]

    switch = True

    while True:
        a_fake, _ = next(iterator)

        switch = not switch

        if switch:
            _, b_fake = next(iterator)
        else:
            b_fake = unet.predict(a_fake)

        a_real, b_real = next(iterator)

        fake = numpy.concatenate((a_fake, b_fake), axis = 1)
        real = numpy.concatenate((a_real, b_real), axis = 1)

        batch_x = numpy.concatenate((fake, real), axis = 0)

        batch_y = numpy.ones((batch_x.shape[0], 1) + discriminator_out_size)
        batch_y[fake.shape[0]:] = 0

        yield batch_x, batch_y

def discriminator_only_generator(iterator, discriminator):
    discriminator_out_size = discriminator.output_shape[-2:]

    while True:
        a_fake, _ = next(iterator)

        _, b_fake = next(iterator)

        a_real, b_real = next(iterator)

        fake = numpy.concatenate((a_fake, b_fake), axis = 1)
        real = numpy.concatenate((a_real, b_real), axis = 1)

        batch_x = numpy.concatenate((fake, real), axis = 0)

        batch_y = numpy.ones((batch_x.shape[0], 1) + discriminator_out_size)
        batch_y[fake.shape[0]:] = 0

        yield batch_x, batch_y


def unet_generator(iterator):
    while True:
        batch = next(iterator)

        yield batch


def pix2pix_generator(iterator, discriminator):
    discriminator_out_size = discriminator.output_shape[-2:]

    for a, b in iterator:
        y = numpy.zeros((a.shape[0], 1) + discriminator_out_size)

        yield [a, b], y


def get_value(x, y, channel, array):
    x1 = x
    y1 = y

    if x1 > 511:
        x1 = 511

    if y1 > 511:
        y1 = 511

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    return array[channel][x1][y1]


def gradient_for_pixel(x1, y1, x2, y2, array):
    r = abs(get_value(x1, y1, 0, array) - get_value(x2, y2, 0, array))
    g = abs(get_value(x1, y1, 1, array) - get_value(x2, y2, 1, array))
    b = abs(get_value(x1, y1, 2, array) - get_value(x2, y2, 2, array))

    return (r + g + b) / 3.0


def gradient_for_pixel_x(x, y, array):
    return gradient_for_pixel(x, y, x + 1, y, array)


def gradient_for_pixel_y(x, y, array):
    return gradient_for_pixel(x, y, x, y + 1, array)


def gradient_for_pixel_d(x, y, array):
    return gradient_for_pixel(x, y, x + 1, y + 1, array)


def to_gradients(array):
    for x in range(0, 512):
        for y in range(0, 512):
            r = gradient_for_pixel_x(x, y, array)
            g = gradient_for_pixel_y(x, y, array)
            b = gradient_for_pixel_d(x, y, array)

            array[0][x][y] = r
            array[1][x][y] = g
            array[2][x][y] = b

    return array