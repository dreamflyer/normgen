from keras.optimizers import Adam

import utils
import model
import callbacks
import report

SAMPLE_TRAIN_PATH = "/Users/dreamflyer/Desktop/diffusetox/train_examples/train"
SAMPLE_VAL_PATH = "/Users/dreamflyer/Desktop/dtox/validation"
SAMPLE_TEST_PATH = "/Users/dreamflyer/Desktop/diffusetox/train_examples/test"

UNET_LOGS_PATH = "logs/unet/loss.bin"

TRAIN_STEPS = 5

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

input_config = ImageDataConfig(size = 512, is_grayscale = False, is_binary = False)
output_config = ImageDataConfig(size = 512, is_grayscale = False, is_binary = False)

def train_iteration(unet, generator, callbacks):
    unet.fit_generator(generator, epochs = 1, steps_per_epoch = TRAIN_STEPS, verbose = False, callbacks = callbacks, workers = 1, use_multiprocessing = False)

def main(resume):
    train_iterator = utils.SimpleFilesIterator(SAMPLE_TRAIN_PATH, 5, input_config, output_config)
    validation_iterator = utils.SimpleFilesIterator(SAMPLE_VAL_PATH, 5, input_config, output_config)

    gen_train = utils.unet_generator(train_iterator)
    gen_val = utils.unet_generator(validation_iterator)

    unet = model.get_unet_model(input_config, output_config, 64, opt = Adam(lr = 0.0003, beta_1 = 0.5))

    count = 0

    cb = callbacks.get_callbacks();

    reporter = report.Reporter(UNET_LOGS_PATH, output_config, resume)

    for epoch in range(100000):
        train_iteration(unet, gen_train, cb)

        u_input, u_fake = validation_iterator.next()
        u_output = unet.predict_on_batch(u_input)

        reporter.handle(u_input, u_fake, u_output)

        count = count + 1

main(False)


