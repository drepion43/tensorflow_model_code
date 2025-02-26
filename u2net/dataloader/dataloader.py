from typing import Dict, Tuple, Any, Text, Optional

import tensorflow as tf

from dataloader.tf_example_decoder import TfExampleDecoder

T = tf.Tensor

class InputReader(object):
    def __init__(self,
                 file_pattern: Text,
                 is_training: bool,
                 batch_size: int,
                 scale: int,
                 buffer_size: int,
                 debug: Optional[bool] = False):
        
        self._file_pattern = file_pattern
        self._is_training = is_training
        self._batch_size = batch_size
        self._scale = scale
        self._buffer_size = buffer_size
        self._debug = debug
        

    @tf.autograph.experimental.do_not_convert
    def dataset_parser(self,
                       value: T,
                       example_decoder: object) -> Tuple[Any, Any]:
        with tf.name_scope('parser'):
            data = example_decoder.decode(value)
            train = data.pop('image')
            label = data.pop('label')
            
            scale = tf.constant(value=[self._scale],
                                dtype=tf.float32)
            data['image'] = tf.divide(x=train,
                                      y=scale)
            label = tf.divide(x=label,
                              y=scale)
        
            return data, label
        

    @property
    def dataset_options(self) -> object:
        options = tf.data.Options()
        options.deterministic = self._debug or not self._is_training
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        return options
    
    def __call__(self) -> object:
        example_decoder = TfExampleDecoder()
        
        dataset = tf.data.Dataset.list_files(self._file_pattern,
                                             shuffle=self._is_training)
        dataset = dataset.interleave(lambda file_name: tf.data.TFRecordDataset(file_name).prefetch(tf.data.AUTOTUNE),
                                     cycle_length=5,
                                     num_parallel_calls=tf.data.AUTOTuNE)
        dataset = dataset.with_options(self.dataset_options)
        dataset = dataset.map(lambda value: self.dataset_parser(value, example_decoder),
                              num_parallel_calls=tf.data.AUTOTUNE)
        
        if self._is_training:
            dataset = dataset.shuffle(self._buffer_size)
            
        dataset = dataset.prefetch(self._batch_size)
        dataset = dataset.batch(self._batch_size,
                                drop_remainder=self._is_training)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset

        