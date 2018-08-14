import keras
import plot
import numpy

class TrainReporter(keras.callbacks.Callback):
    def __init__(self, name):
        self.epoch = 0
        self.batch = 0

        self.name = name

    def on_epoch_begin(self, epoch, logs = {}):
        print("starting epoch: " + str(self.epoch) + ", for model: " + self.name)

    def on_epoch_end(self, epoch, logs = {}):
        print("end epoch: " + str(self.epoch) + ", for model: " + self.name + ", loss: " + str(logs['loss']))

        self.epoch = self.epoch + 1

    def on_batch_begin(self, batch, logs = {}):
        print("epoch: " + str(self.epoch) + ", starting batch: " + str(self.batch) + ", for model: " + self.name)

        self.batch = self.batch + 1

class LossReporter(keras.callbacks.Callback):
    def __init__(self, name):
        self.epoch = 0
        self.batch = 0

        self.plot = plot.SimplePlot(name)

        self.loss = []

        self.name = name

    def on_epoch_end(self, epoch, logs = {}):
        self.loss.append(numpy.asscalar(logs['loss']))

        self.plot.update(self.loss)

        self.epoch = self.epoch + 1


def get_callbacks():
    report_callback = TrainReporter("unet")

    return [report_callback, LossReporter("unet")]

def get_p2p_callbacks(unet, UNET_WEIGHTS_PATH):
    report_callback = TrainReporter("p2p")

    return [report_callback]

def get_discriminator_callbacks(DISCRIMINATOR_WEIGHTS_PATH):
    report_callback = TrainReporter("disc")

    return [report_callback, LossReporter("disc")]