from typing import Text

import tensorflow as tf

T = tf.Tensor

class DenormalizeRootMeanSquaredError(tf.keras.metrics.Metric):
    def __init__(self,
                 denormlization_scale: float = 1.0,
                 name: Text = 'denormalize_root_mean_square_error',
                 **kwargs):
        super(DenormalizeRootMeanSquaredError, self).__init__(name=name, **kwargs)
        self.denormlization_scale = denormlization_scale
        self.total_mse = self.add_weights(name='total_mae',
                                          initializer='zeros')
        self.total_samples = self.add_weights(name='total_sample',
                                              initializer='zeros')
        
    def update_state(self,
                     y_true: T,
                     y_pred: T,
                     sample_weight=None):
        y_true = tf.multiply(x=y_true,
                             y=self.denormlization_scale)
        y_pred = tf.multiply(x=y_pred,
                             y=self.denormlization_scale)
        
        error = tf.subtract(x=y_true,
                            y=y_pred)
        mse = tf.reduce_sum(tf.square(error))
        
        total_count = tf.shape(y_true)[0]
        total_count = tf.cast(total_count,
                              dtype=tf.float32)
        
        self.total_mse.assign_add(mse)
        self.total_samples.assign_add(total_count)
        
    def result(self) -> T:
        result = tf.divide(x=self.total_mse,
                           y=self.total_samples)
        result = tf.sqrt(result)
        return result