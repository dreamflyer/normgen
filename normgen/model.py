import keras

import numpy

from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Activation, Dropout
from keras.layers.merge import Concatenate
from keras.models import Model

from keras import losses

keras.backend.set_image_dim_ordering('th')

def Convolution(num_of_filters, kernel_input_size = 3, strides = 2, border_mode = 'same', **kwargs):
    return Convolution2D(num_of_filters, kernel_size=(kernel_input_size, kernel_input_size), padding = border_mode, strides = (strides, strides), **kwargs)

def BatchNorm(axis = 1, **kwargs):
    return BatchNormalization(axis = axis, **kwargs)

def Deconvolution(filters, output_dim, kernel_size = 2, strides = 2, **kwargs):
    return Conv2DTranspose(filters, kernel_size = (kernel_size, kernel_size), strides = (strides, strides), data_format = 'channels_first', **kwargs)

def merge_layers(inputs, concat_axis = 1):
    return Concatenate(axis = concat_axis)(inputs)

def binary_crossentropy(y_true, y_pred):
    return keras.backend.mean(keras.backend.binary_crossentropy(y_pred, y_true), axis = -1)

def get_unet_model(input_config, output_config, filters, opt, name = 'unet'):
    input = Input(shape = input_config.shape)

    #512 x 512
    conv1 = Convolution(filters)(input)

    conv1 = BatchNorm()(conv1)

    conv1_2 = conv1

    x = LeakyReLU(0.2)(conv1)
    #n x 256 x 256

    conv2 = Convolution(filters * 2)(x)
    conv2 = BatchNorm()(conv2)
    x = LeakyReLU(0.2)(conv2)
    #2n x 128 x 128

    conv3 = Convolution(filters * 4)(x)
    conv3 = BatchNorm()(conv3)
    x = LeakyReLU(0.2)(conv3)
    #4n x 64 x 64

    conv4 = Convolution(filters * 8)(x)
    conv4 = BatchNorm()(conv4)
    x = LeakyReLU(0.2)(conv4)
    #8n x 32 x 32

    conv5 = Convolution(filters * 8)(x)
    conv5 = BatchNorm()(conv5)
    x = LeakyReLU(0.2)(conv5)
    #8n x 16 x 16

    conv6 = Convolution(filters * 8)(x)
    conv6 = BatchNorm()(conv6)
    x = LeakyReLU(0.2)(conv6)
    #8n x 8 x 8

    conv7 = Convolution(filters * 8)(x)
    conv7 = BatchNorm()(conv7)
    x = LeakyReLU(0.2)(conv7)
    #8n x 4 x 4

    conv8 = Convolution(filters * 8)(x)
    conv8 = BatchNorm()(conv8)
    x = LeakyReLU(0.2)(conv8)
    #8n x 2 x 2

    conv9 = Convolution(filters * 8, 2, 1, 'valid')(x)
    conv9 = BatchNorm()(conv9)
    x = LeakyReLU(0.2)(conv9)
    # nf*8 x 1 x 1

    dconv1 = Deconvolution(filters * 8, 2, 2, 1)(x)
    dconv1 = BatchNorm()(dconv1)
    dconv1 = Dropout(0.5)(dconv1)
    x = merge_layers([dconv1, conv8])
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 2 x 2

    dconv2 = Deconvolution(filters * 8, 4)(x)
    dconv2 = BatchNorm()(dconv2)
    dconv2 = Dropout(0.5)(dconv2)
    x = merge_layers([dconv2, conv7])
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 4 x 4

    dconv3 = Deconvolution(filters * 8, 8)(x)
    dconv3 = BatchNorm()(dconv3)
    dconv3 = Dropout(0.5)(dconv3)
    x = merge_layers([dconv3, conv6])
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 8 x 8

    dconv4 = Deconvolution(filters * 8, 16)(x)
    dconv4 = BatchNorm()(dconv4)
    x = merge_layers([dconv4, conv5])
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 16 x 16

    dconv5 = Deconvolution(filters * 8, 32)(x)
    dconv5 = BatchNorm()(dconv5)
    x = merge_layers([dconv5, conv4])
    x = LeakyReLU(0.2)(x)
    # nf*(8 + 8) x 32 x 32

    dconv6 = Deconvolution(filters * 4, 64)(x)
    dconv6 = BatchNorm()(dconv6)
    x = merge_layers([dconv6, conv3])
    x = LeakyReLU(0.2)(x)
    # nf*(4 + 4) x 64 x 64

    dconv7 = Deconvolution(filters * 2, 128)(x)
    dconv7 = BatchNorm()(dconv7)
    x = merge_layers([dconv7, conv2])
    x = LeakyReLU(0.2)(x)
    # nf*(2 + 2) x 128 x 128

    dconv8 = Deconvolution(filters, 256)(x)
    dconv8 = BatchNorm()(dconv8)
    x = merge_layers([dconv8, conv1])
    x = LeakyReLU(0.2)(x)
    # nf*(1 + 1) x 256 x 256

    dconv9 = Deconvolution(output_config.shape[0], output_config.size)(x)
    # out_ch x 512 x 512

    act = 'sigmoid' if output_config.is_binary else 'tanh'
    out = Activation(act)(dconv9)

    unet = Model(input, out, name = name)

    unet.compile(optimizer = opt, loss = d_loss)

    return unet

def get_discriminator(input_config, output_config, filters, opt, name = 'd'):
    input = Input(shape = (input_config.shape[0] + output_config.shape[0], input_config.size, input_config.size))

    # (a_ch + b_ch) x 512 x 512
    conv1 = Convolution(filters)(input)
    x = LeakyReLU(0.2)(conv1)
    # nf x 256 x 256

    conv2 = Convolution(filters * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 128 x 128

    conv3 = Convolution(filters * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 64 x 64

    conv4 = Convolution(filters * 8)(x)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 32 x 32

    conv5 = Convolution(1)(x)
    out = Activation('sigmoid')(conv5)
    # 1 x 16 x 16

    d = Model(input, out, name = name)

    def d_loss(y_true, y_pred):
        return keras.backend.mean(keras.backend.abs(y_true - y_pred))

        #return binary_crossentropy(keras.backend.batch_flatten(y_true), keras.backend.batch_flatten(y_pred))

    d.compile(optimizer = opt, loss = d_loss)

    return d

def d_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.abs(y_true - y_pred))

def get_pix2pix(unet, discriminator, input_config, output_config, opt, alpha = 100, name = 'pix2pix'):
    input = Input(shape = input_config.shape)
    output = Input(shape = output_config.shape)

    unet_output = unet(input)

    discriminator_input = merge_layers([input, unet_output])

    def pix2pix_loss(y_true, y_pred):
        y_true_flat = keras.backend.batch_flatten(y_true)
        y_pred_flat = keras.backend.batch_flatten(y_pred)

        L_adv = binary_crossentropy(y_true_flat, y_pred_flat)

        output_flat = keras.backend.batch_flatten(output)
        unet_output_flat = keras.backend.batch_flatten(unet_output)

        if output_config.is_binary:
            L_atob = binary_crossentropy(output_flat, unet_output_flat)
        else:
            L_atob = keras.backend.mean(keras.backend.abs(output_flat - unet_output_flat))

        return L_adv + alpha * L_atob