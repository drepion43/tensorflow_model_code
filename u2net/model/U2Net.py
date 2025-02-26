from typing import Text, Optional, Dict, List, Tuple
from param_config import get_config

import tensorflow as tf
from collections import deque

from model.model_utils.u2net_utils import RSU, _upsample_like

T = tf.Tensor

class U2NetModuel(tf.keras.Model):
    def __init__(self,
                 output_channels: int = 3,
                 config: Optional[Dict] = None,
                 model_type: Text = 'U2Net',
                 name: Text = 'U2Net',
                 **kwargs):
        super(U2NetModuel, self).__init__(name=name, **kwargs)
        self.output_channels = output_channels
        
        self.config = config or get_config()
        self.encoder_config = self.config['model'][model_type]['U2NetEncoder']
        
        self.decoder_config = self.config['model'][model_type]['U2NetDecoder']
        self.module_size = len(self.decoder_config.items())
        
        additional_side_list = deque()
        for idx, (k, v) in enumerate(self.encoder_config.items()):
            _, _, _, _, _, side = v
            if side != -1:
                additional_side_list.appendleft(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=side,
                                               kernel_size=(3, 3),
                                               padding='same',
                                               name=name + f"__SideConv{idx}")
                    ], name=name + f"__EncoderSideStage_{idx}")
                )
        
        self.side_list = deque()
        for idx in range(self.module_size):
            self.side_list.appendleft(
                tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters=self.output_channels,
                                           kernel_size=(3, 3),
                                           padding='same',
                                           name=name + f"__SideConv{idx}")
                ], name=name + f"__DecoderSideStage_{idx}")
            )
        self.side_list.extend(additional_side_list)

        self.scale_in = U2NetEncoder(config=config,
                                     model_type=model_type,
                                     name=name + "__U2NetEncoder")
        
        self.scale_out = U2NetDecoder(config=config,
                                     model_type=model_type,
                                     name=name + "__U2NetDecoder")
        self.outconv = tf.keras.layers.Conv2(filters=self.output_channels,
                                             kernel_size=(1, 1),
                                             padding="same",
                                             name=name + f"__OutConv")
        
    def call(self,
             inputs: Dict,
             training: bool) -> Tuple[T, List]:
        
        image = inputs['image']
        # Encoder
        encoder_output, encoder_residuals = self.scale_in(inputs=image,
                                                          training=training)
        
        # Decoder
        decoder_output, decoder_residuals = self.scale_out(inputs=encoder_output,
                                                           encoder_residuals=encoder_residuals,
                                                           training=training)
        
        # side output
        side_residuals = []
        for idx, side_layer in enumerate(self.side_list):
            decoder_x = decoder_residuals[idx]
            side_output = side_layer(decoder_x,
                                     training=training)
            if idx == 0:
                stem_output = tf.identity(input=side_output)
                side_residuals.append(side_output)
            else:
                side_output = _upsample_like(src=side_output,
                                             tar=stem_output)
                side_residuals.append(side_output)
            
        output = self.outconv(tf.concat(side_residuals,
                                        axis=-1),
                              training=training)
        
        return output, side_residuals
        
class U2NetEncoder(tf.keras.Model):
    def __init__(self,
                 config: Optional[Dict] = None,
                 model_type: Text = 'U2Net',
                 name: Text = 'U2NetEncoder'):
        super(U2NetEncoder, self).__init__(name=name)
        
        config = config or get_config()
        self.config = config
        self.encoder_config = self.config['model'][model_type]['U2NetEncoder']
        self.module_size = len(self.encoder_config.items())
        
        self.encoder_list = []
        self.pool_list = []
        for idx, (k, v) in enumerate(self.encoder_config.items()):
            layer_name = k
            rsu_num, in_ch, mid_ch, out_ch, dilated, side = v
            self.encoder_list.append(
                tf.keras.Sequential([
                    RSU(rsu_num=rsu_num,
                        is_FRES=dilated,
                        output_channels=out_ch,
                        mid_channels=mid_ch,
                        name=name + f"__{layer_name}")
                ], 
                                    name=name + f"__EncoderStage_{idx}")
            )
            if self.module_size - 1 > idx:
                self.pool_list.append(
                    tf.keras.Sequential([
                        tf.keras.layer.MaxPool2D(pool_size=(2, 2),
                                                 strides=2,
                                                 padding='same',
                                                 name=name + f"__MaxPool_{idx}")
                    ], 
                                        name=name + f"__MaxPoolSequential_{idx}")
                )
        
    def call(self,
             inputs: T,
             training: bool) -> Tuple[T, deque]:
        x = tf.identity(input=inputs)
        
        encoder_residuals = deque()
        for idx, encoder_layer in enumerate(self.encoder_list):
            x = encoder_layer(x,
                              training=training)
            encoder_residuals.appendleft(x)
            if idx < self.module_size - 1:
                x = self.pool_list[idx](x,
                                        training=training)
                
        output = _upsample_like(src=encoder_residuals[0],
                                tar=encoder_residuals[1])

        return output, encoder_residuals
    

class U2NetDecoder(tf.keras.Model):
    def __init__(self,
                 config: Optional[Dict] = None,
                 model_type: Text = 'U2Net',
                 name: Text = 'U2NetDecoder'):
        super(U2NetDecoder, self).__init__(name=name)
        
        config = config or get_config()
        self.config = config
        self.encoder_config = self.config['model'][model_type]['U2NetEncoder']
        
        self.decoder_config = self.config['model'][model_type]['U2NetDecoder']

        self.module_size = len(self.decoder_config.items())
        
        additional_side_list = deque()
        for idx, (k, v) in enumerate(self.encoder_config.items()):
            _, _, _, _, _, side = v
            if side != -1:
                additional_side_list.appendleft(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=side,
                                               kernel_size=(3, 3),
                                               padding='same',
                                               name=name + f"__SideConv{idx}")
                    ], name=name + f"__EncoderSideStage_{idx}")
                )
        
        self.decoder_list = []
        self.side_list = deque()
        
        for idx, (k, v) in enumerate(self.decoder_config.items()):
            layer_name = k
            rsu_num, in_ch, out_ch, mid_ch, dilated, side = v
            self.decoder_list.append(
                tf.keras.Sequential([
                    RSU(rsu_num=rsu_num,
                        is_FRES=dilated,
                        output_channels=out_ch,
                        mid_channels=mid_ch,
                        name=name + f"__{layer_name}")
                ], name=name + f"__DecoderStage_{idx}")
            )
            self.side_list.appendleft(
                    tf.keras.Sequential([
                        tf.keras.layers.Conv2D(filters=side,
                                               kernel_size=(3, 3),
                                               padding='same',
                                               name=name + f"__SideConv{idx}")
                    ], name=name + f"__DecoderSideStage_{idx}")
                )
        self.side_list.extend(additional_side_list)
        
    def call(self,
             inputs: T,
             encoder_residuals: deque,
             training: bool) -> Tuple[T, deque]:
        x = tf.identity(input=inputs)
        
        decoder_residuals = deque([x])
        for idx, decoder_layer in enumerate(self.decoder_list):
            encoder_x = encoder_residuals[idx + 1]
            x = decoder_layer(tf.concat([x, encoder_x],
                                        axis=-1),
                              training=training)
            decoder_residuals.appendleft(x)
            if idx < self.module_size - 1:
                x = _upsample_like(src=x,
                                   tar=encoder_residuals[idx + 2])
        
        decoder_output = tf.identity(input=x)
        
        return decoder_output, decoder_residuals