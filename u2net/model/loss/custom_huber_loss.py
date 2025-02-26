from typing import List, Tuple

import tensorflow as tf

T = tf.Tensor


class CustomHuberLoss(tf.keras.losses.Loss):
    def __init__(self,
                 delta: float = 1.0):
        super(CustomHuberLoss, self).__init__()
        self.delta = delta
        
    def calc_loss(self,
                  y_true: T,
                  y_pred: T) -> T:
        error = y_true - y_pred
        is_smaller_error = tf.abs(error) <= self.delta
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.delta * (tf.abs(error) - (0.5 * self.delta))
        
        huber_loss = tf.where(is_smaller_error, small_error_loss, big_error_loss)
        
        return huber_loss
    
    def call(self,
             y_true: T,
             y_pred_0: T,
             y_pred_list: List) -> Tuple[T, T]:
        
        loss0 = self.calc_loss(y_true=y_true,
                               y_pred=y_pred_0)
    
        total_loss = [loss0]
        
        for y_pred in y_pred_list:
            total_loss.append(self.calc_loss(y_true=y_true,
                                             y_pred=y_pred))
            
        loss = tf.reduce_sum(tf.stack(total_loss))
        
        return tf.reduce_sum(loss0), loss
    