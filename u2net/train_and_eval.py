import os

import shutil
import textwrap

import tensorflow as tf
import numpy as np


from typing import List, Dict, Text
from glob import glob

T = tf.Tensor

import param_config
from model.U2Net import U2NetModuel
from dataloader.dataloader import InputReader
from logging_info import LoggingConfig
from model.loss.custom_huber_loss import CustomHuberLoss
from model.metrics.metrics import DenormalizeRootMeanSquaredError
logger = LoggingConfig.get_logger(__name__)

class CustomTrainLoop(object):
    def __init__(self,
                 model: object,
                 **kwargs):
        self.model = model
        
    @tf.function
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()
            
    @tf.function(input_signature=[
        {
            'image': tf.TensorSpec(shape=(None, 416, 416, 3), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(None, 416, 416, 3), dtype=tf.float32)
    ])
    def train_step(self,
                   inputs: Dict,
                   targets: T) -> T:
        with tf.GradientTape() as tape:
            predictions, predictions_list = self.model(inputs=inputs,
                                                       training=True)
            loss0, loss = self.loss_fn(y_true=targets,
                                       y_pred_0=predictions,
                                       y_pred_list=predictions_list)
        grads = tape.gradient(target=loss,
                              sources=self.model.trainalbe_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        for metric in self.metrics:
            metric.update_state(y_true=targets,
                                y_pred=predictions)
        return loss0, loss
    
    @tf.function(input_signature=[
        {
            'image': tf.TensorSpec(shape=(None, 416, 416, 3), dtype=tf.float32)
        },
        tf.TensorSpec(shape=(None, 416, 416, 3), dtype=tf.float32)
    ])
    def valid_step(self,
                   inputs: Dict,
                   targets: T) -> T:
        predictions, predictions_list = self.model(inputs=inputs,
                                                    training=False)
        loss0, loss = self.loss_fn(y_true=targets,
                                    y_pred_0=predictions,
                                    y_pred_list=predictions_list)

        for metric in self.metrics:
            metric.update_state(y_true=targets,
                                y_pred=predictions)
        return loss0, loss
    
    def train(self,
              train_file_pattern: Text,
              valid_file_pattern: Text,
              optimizer: object,
              metrics: List,
              history: Dict,
              args_config: Dict):
        
        epochs = args_config['epochs']
        patience_limit = args_config['patience_limit']
        batch_size = args_config['batch_size']
        scale = args_config['scale']
        ckpt_path = args_config['ckpt_path']
        delta = args_config['delta']
        
        
        self.loss_fn = CustomHuberLoss(delta=delta)
        
        train_dataset = InputReader(file_pattern=train_file_pattern,
                                    is_training=True,
                                    scale=scale,
                                    batch_size=batch_size)
        valid_dataset = InputReader(file_pattern=valid_file_pattern,
                                    is_training=False,
                                    scale=scale,
                                    batch_size=batch_size)
        
        inputs, targets = next(iter(valid_dataset))
        
        self.optimizer = optimizer
        self.metrics = metrics
        _ = self.model(inputs=inputs, training=False)
        print(self.model.summary())
        logger.info(self.model.summary())
        
        train_step_per_epoch = -1
        valid_step_per_epoch = -1
        
        train_loss = float('inf')
        valid_loss = float('inf')
        best_valid_loss = float('inf')
        patience = 0
        
        directory = os.path.dirname(ckpt_path)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         model=self.model)
        manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                             directory=directory,
                                             max_to_keep=epochs)
        
        for epoch in range(epochs):
            print('Epoch %s/%s' % (epoch + 1, epochs))
            logger.info('Epoch %s/%s' % (epoch + 1, epochs))
            
            self.reset_metrics()
            
            # train
            train_step_cnt = 0
            train_loss_sum = 0
            train_loss_0_sum = 0
            
            for step, (inputs, y_true) in enumerate(train_dataset.__iter__()):
                train_loss0, train_loss = self.train_step(inputs=inputs,
                                                          targets=y_true)
                train_loss_0_sum += train_loss0.numpy()
                train_loss_sum += train_loss.numpy()
                train_step_cnt += 1
                
                train_loss = train_loss_sum / train_step_cnt
                train_loss_0 = train_loss_0_sum / train_step_cnt
                
                terminal_size = shutil.get_terminal_size()
                
                if epoch == 0:
                    train_step_per_epoch = step
                    train_result = '%s/Unknown' % step
                else:
                    train_result = '%s/%s' % (step, train_step_per_epoch)
                    
                train_result += ' - loss: %e' % (train_loss)
                train_result += ' - loss0: %e' % (train_loss_0)
                train_result += ''.join(' - %s: %.4f' % (metric.name, metric.result().numpy()) for metric in self.metrics)
                train_result = textwrap.wrap(text=train_result,
                                             width=terminal_size[0])
                
                num_lines = len(train_result)
                train_result = '\n'.join(train_result)
                print(train_result)
                print('\033[F' * num_lines, end='', flush=True)
                
            print()
            logger.info(train_result)
            
            # valid
            valid_step_cnt = 0
            valid_loss_sum = 0
            valid_loss_0_sum = 0
            
            self.reset_metrics()
            print('Valid Start.')
            logger.info('Valid Start.')
            
            for step, (inputs, y_true) in enumerate(valid_dataset.__iter__()):
                valid_loss0, valid_loss = self.valid_step(inputs=inputs,
                                                          targets=y_true)
                valid_loss_0_sum += valid_loss0.numpy()
                valid_loss_sum += valid_loss.numpy()
                valid_step_cnt += 1
                
                valid_loss = valid_loss_sum / valid_step_cnt
                valid_loss_0 = valid_loss_0_sum / valid_step_cnt
                
                terminal_size = shutil.get_terminal_size()
                
                if epoch == 0:
                    valid_step_per_epoch = step
                    valid_result = '%s/Unknown' % step
                else:
                    valid_result = '%s/%s' % (step, valid_step_per_epoch)
                    
                valid_result += ' - loss: %e' % (valid_loss)
                valid_result += ' - loss0: %e' % (valid_loss_0)
                valid_result += ''.join(' - %s: %.4f' % (metric.name, metric.result().numpy()) for metric in self.metrics)
                valid_result = textwrap.wrap(text=valid_result,
                                             width=terminal_size[0])
                
                num_lines = len(valid_result)
                valid_result = '\n'.join(valid_result)
                print(valid_result)
                print('\033[F' * num_lines, end='', flush=True)
                
            print(valid_result)
            logger.info(valid_result)
            
            print()
            
            # early stopping
            if patience > patience_limit:
                break
            
            if not tf.io.gfile.isdir(directory):
                tf.io.gfile.mkdir(directory)
            
            if valid_loss < best_valid_loss:
                save_path = manager.save()
                patience = 0
                print('Epoch %s: val_loss improved from %s to %s, saving model to %s' %
                      (epoch + 1, best_valid_loss, valid_loss, save_path))
                logger.info('Epoch %s: val_loss improved from %s to %s, saving model to %s' %
                      (epoch + 1, best_valid_loss, valid_loss, save_path))
                best_valid_loss = valid_loss
            else:
                patience += 1
                
            print()
            

if __name__ == '__main__':
    train_file_pattern = ""
    valid_file_pattern = ""

    ckpt_path = './ckpt/'

    args_coinfig = {
        'epochs': 10,
        'patience_limit': 5,
        'batch_size': 128,
        'scale': 255,
        'ckpt_path': ckpt_path,
        'delta': 10 / 255
    }
    
    model = U2NetModuel(model_type='U2Net')
    # model = U2NetModuel(model_type='U2NetLite')

    trainer = CustomTrainLoop(model=model)
    trainer.train(train_file_pattern=train_file_pattern,
                  valid_file_pattern=valid_file_pattern,
                  optimizer=tf.keras.optimizer.Adam(learning_rate=1e-4),
                  metrics=[
                      tf.keras.metrics.MeanAbsoluteError(
                          name='mean_absolute_error',
                          dtype=None
                          ),
                      DenormalizeRootMeanSquaredError(denormlization_scale=255)
                  ],
                  args_config=args_coinfig)