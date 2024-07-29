# %% [code]
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D,  UpSampling2D, BatchNormalization, Activation, Add, LayerNormalization)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, skip='identity', disc=False, param={}, skip_param={}, **kwargs):
        super().__init__()
        param_dict = param.copy()
        skip_kwargs = skip_param.copy()
        activation = param_dict.pop('activation', None)
        if activation is None: activation = 'relu'

        if skip == 'identity' or skip == 'linear':
            self.skip_connection = Activation('linear')
            self.conv1 = Conv2D(filters, (3,3), (1,1), 'same', use_bias=False, **param_dict)
            self.conv2 = Conv2D(filters, (3,3), (1,1), 'same', use_bias=True, **param_dict)
        elif skip == 'down':
            self.skip_connection = Conv2D(filters, (1,1), (2,2), 'same', use_bias=True, **skip_kwargs)
            self.conv1 = Conv2D(filters, (3,3), (1,1), 'same', use_bias=False, **param_dict)
            self.conv2 = Conv2D(filters, (3,3), (2,2), 'same', use_bias=True, **param_dict)
        else:
            self.skip_connection = UpConv2D(filters, (1,1), (2,2), 'same', use_bias=True, kwargs=skip_kwargs)
            self.conv1 = UpConv2D(filters, (3,3), (2,2), 'same', use_bias=False, param_dict=param_dict)
            self.conv2 = Conv2D(filters, (3,3), (1,1), 'same', use_bias=True, **param_dict)
            
        if not disc: 
            self.norm1 = BatchNormalization(); self.norm2 = BatchNormalization()
        else: 
            self.norm1 = LayerNormalization(); self.norm2 = LayerNormalization() # LayerNorm for discriminator
            
        self.act1 = Activation(activation)
        self.act2 = Activation(activation)
        self.add = Add()

    def call(self, x):
        res = self.skip_connection(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.add([x, res])
        x = self.norm2(x)
        x = self.act2(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config 

class UpConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), strides=(1,1), padding='valid', use_bias=True, param_dict={}, **kwargs):
        super().__init__()
        self.up1 = UpSampling2D(strides)
        self.conv1 = Conv2D(filters, kernel_size, (1,1), padding, use_bias=use_bias, **param_dict)

    def call(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config 