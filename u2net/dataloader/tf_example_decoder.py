from typing import Dict

import tensorflow as tf
T = tf.Tensor

class TfExampleDecoder(object):
    def __init__(self):
        self._keys_to_features = {
            'image': tf.io.VarLenFeature(tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.int64),
        }
        
    def decode(self,
               serialized_example: T) -> Dict:
        parsed_tensor = tf.io.parse_single_example(serialized=serialized_example,
                                                   features=self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k])
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k])
        
        decoded_tensors = {
            'image': parsed_tensors['image'],
            'label': parsed_tensors['label'],   
        }
        return decoded_tensors