import tensorflow as tf
import tensorflow.keras.backend as K

T = tf.Tensor

class Swish(tf.keras.layers.Activation):
    def __init__(self,
                 activation,
                 **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        
    def call(self,
             x: T) -> T:
        return x * K.sigmoid(x)
