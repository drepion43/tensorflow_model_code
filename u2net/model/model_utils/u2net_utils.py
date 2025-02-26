from typing import Text, Tuple
from collections import deque

import tensorflow as tf

T = tf.Tensor

def _upsample_like(src, tar):
    height = tf.shape(tar)[1]
    width = tf.shape(tar)[2]
    
    src = tf.image.resize(src,
                          size=(height, width),
                          method='bilinear')
    return src

class BasicConv(tf.keras.layers.Layer):
    def __init__(self,
                 output_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation_rate: int = 1,
                 groups: int = 1,
                 bias: bool = False,
                 kernel_initializer: Text = 'he_normal',
                 padding: Text = 'same',
                 eps: float = 1e-5,
                 momentum: float = 0.99,
                 name: Text = 'BasicConv'):
        super(BasicConv, self).__init__(name=name)
        
        self.conv2d = tf.keras.layers.Conv2D(filters=output_channels,
                                             kernel_size=kernel_size,
                                             strides=stride,
                                             padding=padding,
                                             dilation_rate=dilation_rate,
                                             groups=groups,
                                             use_bias=bias,
                                             kernel_initializer=kernel_initializer,
                                             name=name + "__Conv")
        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=eps,
                                                             momentum=momentum,
                                                             name=name + "__BatchNorm")
        self.relu = tf.keras.layers.Activation('relu')
        
    def call(self,
             inputs: T,
             training: bool) -> T:
        output = self.conv2d(inputs, training=training)
        output = self.batch_norm(output, training=training)
        output = self.relu(output, training=training)
        
        
class RSU(tf.keras.Model):
    def __init__(self,
                 rsu_num: int = 7,
                 is_FRES: bool = False,
                 output_channels: int = 3,
                 mid_channels: int = 12,
                 name: Text = 'RSU'):
        super(RSU, self).__init__(name=name)
        self.is_FRES = is_FRES
        self.rsu_num = rsu_num
        
        self.rebconvin = BasicConv(output_channels=output_channels,
                                   dilation_rate=(1, 1),
                                   name=name + "__RECONVIN")
        
        # rebconv
        self.rebconv_list = []
        self.pool_list = []
        for idx in range(1, self.rsu_num - 1):
            if self.is_FRES:
                dirate = (pow(2, idx - 1), pow(2, idx - 1))
            else:
                dirate = (1, 1)
                
            self.rebconv_list.append(
                tf.keras.Sequential([
                    BasicConv(output_channels=mid_channels,
                              dilation_rate=dirate,
                              name=name + f"__ENCODER_RECONV_{idx}")
                ],
                                    name=name + f"__ENCODER")
            )
            
            if not self.is_FRES:
                self.pool_list.append(
                    tf.keras.Sequential([
                        tf.keras.layers.MaxPool2D(pool_size=(2, 2),
                                                  strides=2,
                                                  padding='same',
                                                  name=name + f"__ENCODER_MaxPool_{idx}")
                    ],
                                        name=name + f"__ENCODER_POOL")
                )
        
        if self.is_FRES:
            self.reconv_x = BasicConv(output_channels=mid_channels,
                                    dilation_rate=(pow(2, self.rsu_num - 2), pow(2, self.rsu_num - 2)),
                                    name=name + f"__ENCODER_RECONV_{self.rsu_num - 2}")
            self.reconv_xx = BasicConv(output_channels=mid_channels,
                                    dilation_rate=(pow(2, self.rsu_num - 1), pow(2, self.rsu_num - 1)),
                                    name=name + f"__ENCODER_RECONV_{self.rsu_num - 1}")
        else:
            self.reconv_x = BasicConv(output_channels=mid_channels,
                                    dilation_rate=(1, 1),
                                    name=name + f"__ENCODER_RECONV_{self.rsu_num - 2}")
            self.reconv_xx = BasicConv(output_channels=mid_channels,
                                    dilation_rate=(2, 2),
                                    name=name + f"__ENCODER_RECONV_{self.rsu_num - 1}")
    
        self.rebconvd_list = []
        for idx in range(self.rsu_num - 1, 1, -1):
            if self.is_FRES:
                dirate = (pow(2, idx - 1), pow(2, idx - 1))
            else:
                dirate = (1, 1)
            
            self.rebconvd_list.append(
                tf.keras.Sequential([
                    BasicConv(output_channels=mid_channels,
                            dilation_rate=dirate,
                            name=name + f"__DECODER_RECONVD_{idx}")
                ], 
                                    name=name + f"__DECODER")
            )
        self.reconv1d = BasicConv(output_channels=output_channels,
                                dilation_rate=(1, 1),
                                name=name + "__DECODER_RECONVD_1")
    
    def call(self,
             inputs: T,
             training: bool) -> T:
        hx = tf.identity(input=inputs)
        hxin = self.rebconvin(hx, training=training)
        x = tf.identity(input=hxin)
        
        # reconv
        encoder_outputs = deque()
        for idx in range(self.rsu_num - 2):
            x = self.rebconv_list[idx](x,
                                       training=training)
            encoder_outputs.appendleft(x)
            
            if not self.is_FRES:
                x = self.pool_list[idx](x,
                                        training=training)
        x = self.reconv_x(x,
                          training=training)
        xx = self.reconv_xx(x,
                            training=training)
        
        x1, x2 = tf.identity(input=x), tf.identity(input=xx)

        # UpSampling
        for idx in range(self.rsu_num - 2):
            x2 = self.rebconvd_list[idx](tf.concat([x2, x1],
                                                   axis=-1),
                                         training=training)
            x1 = tf.identity(input=encoder_outputs[idx])
            
            if not self.is_FRES:
                x2 = _upsample_like(src=x2,
                                    tar=x1)
                
        hx1d = self.reconv1d(tf.concat([x2, x1]),
                             axis=-1)
        
        return hx1d + hxin
    