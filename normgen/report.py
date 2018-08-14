import matplotlib.pyplot as plt
import keras
import _pickle as pickle

import numpy

import plot

import image_util

class Reporter(object):
    def __init__(self, logs_path, output_config, resume = False):
        self.path = logs_path
        self.logs = {'val_loss': []}
        self.output_config = output_config

        self.image = plot.ImagePlot()
        self.plot = plot.SimplePlot("unet")

        if resume:
            self.logs = pickle.load(open(logs_path, 'rb'))
            self.logs['val_loss'] = self.logs['val_loss']


    def handle(self, u_input, u_fake, u_output):
        losses = keras.backend.eval(self.getLosses(u_output, u_fake))

        losses = numpy.asscalar(losses)

        self.logs['val_loss'].append(losses)

        self.image.update(u_input, u_output, u_fake)

        pickle.dump(self.logs, open(self.path, 'wb'))

        self.plot.update(self.logs['val_loss'])

    def getLosses(self, u_output, u_fake):
        if self.output_config.is_binary:
            return binary_crossentropy(keras.backend.batch_flatten(u_output), keras.backend.batch_flatten(u_fake))

        return keras.backend.mean(keras.backend.abs(u_output - u_fake))


def binary_crossentropy(y_true, y_pred):
    return keras.backend.mean(keras.backend.binary_crossentropy(y_pred, y_true), axis = -1)