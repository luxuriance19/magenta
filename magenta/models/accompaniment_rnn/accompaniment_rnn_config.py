# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Helper functions to select an underlying MelodyEncoderDecoder from flags."""

# internal imports
import tensorflow as tf

import magenta
from magenta.models.accompaniment_rnn import accompaniment_rnn_encoder_decoder

FLAGS = tf.app.flags.FLAGS
'''
tf.app.flags.DEFINE_string(
    'config',
    None,
    'Which config to use. Mutually exclusive with `--melody_encoder_decoder`.')
tf.app.flags.DEFINE_string(
    'melody_encoder_decoder',
    None,
    "Which encoder/decoder to use for individual melodies. Must be one of "
    "'onehot', 'lookback', or 'key'. Mutually exclusive with `--config`.")
tf.app.flags.DEFINE_integer(
    'predictahead_steps',
    0,
    'The number of steps ahead of the conditioned melody to output the '
    'predicted melody. Mutually exclusive with `--config`.')
tf.app.flags.DEFINE_string(
    'generator_id',
    None,
    'A unique ID for the generator. Required when `--config` is not supplied. '
    'Overrides the default if `--config` is supplied.')
tf.app.flags.DEFINE_string(
    'generator_description',
    None,
    'A description of the generator. Required when `--config` is not supplied. '
    'Overrides the default if `--config` is supplied.')
tf.app.flags.DEFINE_string(
    'hparams', '{}',
    'String representation of a Python dictionary containing hyperparameter '
    'to value mapping. This mapping is merged with the default '
    'hyperparameters if `--config` is also supplied.')
'''

class AccompanimentRnnConfigException(Exception):
  pass


class AccompanimentRnnConfig(object):
  """Stores a configuration for an AccompanimentRnn.

  Attributes:
    details: The GeneratorDetails message describing the config.
    encoder_decoder: The MelodyPairEncoderDecoder object to use for
        encoding/decoding of MelodyPairs.
    hparams: The HParams containing hyperparameters to use.
  """

  def __init__(self, details, encoder_decoder, hparams):
    self.details = details
    self.encoder_decoder = encoder_decoder
    self.hparams = hparams


# Default configurations.
default_configs = {
    'accompaniment_512x3_p2': AccompanimentRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='accompaniment_512x3_p2',
            description='Accompaniment RNN with lookback encoding, 3 512-node '
                        'layers, and 2-step lookahead.'),
        accompaniment_rnn_encoder_decoder.AccompanimentRnnEncoderDecoder(
            magenta.music.LookbackMelodyEncoderDecoder(),
            predictahead_steps=2),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=5,
            initial_learning_rate=0.01,
            decay_steps=1000,
            decay_rate=0.95)),
    'accompaniment_512x3_p2_notranspose': AccompanimentRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='accompaniment_512x3_p2_notranspose',
            description='Accompaniment RNN with lookback encoding, 3 512-node '
                        'layers, 2-step lookahead, and transpose disabled.'),
        accompaniment_rnn_encoder_decoder.AccompanimentRnnEncoderDecoder(
            magenta.music.LookbackMelodyEncoderDecoder(
                min_note=0, max_note=128, transpose_to_key=None),
            predictahead_steps=2),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97)),
    'accompaniment_att64_512x3_p2': AccompanimentRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='accompaniment_att64_512x3_p2',
            description='Accompaniment RNN with lookback encoding, 3 512-node '
                        'layers, 64-length attention, and 2-step lookahead.'),
        accompaniment_rnn_encoder_decoder.AccompanimentRnnEncoderDecoder(
            magenta.music.LookbackMelodyEncoderDecoder(),
            predictahead_steps=2),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=64,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97)),
    'accompaniment_att64_512x3_p2_notranspose': AccompanimentRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='accompaniment_att64_512x3_p2_notranspose',
            description='Accompaniment RNN with lookback encoding, 3 512-node '
                        'layers, 64-length attention, 2-step lookahead, and '
                        'transpose disabled.'),
        accompaniment_rnn_encoder_decoder.AccompanimentRnnEncoderDecoder(
            magenta.music.LookbackMelodyEncoderDecoder(
                min_note=0, max_note=128, transpose_to_key=None),
            predictahead_steps=2),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=64,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97))
}


# Available MelodyEncoderDecoder classes for encoding the individual melodies.
melody_encoder_decoders = {
    'onehot': magenta.music.OneHotMelodyEncoderDecoder,
    'lookback': magenta.music.LookbackMelodyEncoderDecoder,
    'key': magenta.music.KeyMelodyEncoderDecoder
}


def config_from_flags():
  """Parses flags and returns the appropriate MelodyRnnConfig.

  If `--config` is supplied, returns the matching default MelodyRnnConfig after
  updating the hyperparameters based on `--hparams`.
  If `--melody_encoder_decoder` is supplied, returns a new MelodyRnnConfig using
  the matching MelodyEncoderDecoder, generator details supplied by
  `--generator_id` and `--generator_description`, and hyperparameters based on
  `--hparams`.

  Returns:
     The appropriate MelodyRnnConfig based on the supplied flags.
  Raises:
     AccompanimentRnnConfigException: When not exactly one of `--config` or
         `melody_encoder_decoder` is supplied or the supplied values are
         invalid.
  """
  if (FLAGS.melody_encoder_decoder, FLAGS.config).count(None) != 1:
    raise AccompanimentRnnConfigException(
        'Exactly one of `--config` or `--melody_encoder_decoder` must be '
        'supplied.')

  if FLAGS.melody_encoder_decoder is not None:
    if FLAGS.melody_encoder_decoder not in melody_encoder_decoders:
      raise AccompanimentRnnConfigException(
          '`--melody_encoder_decoder` must be one of %s. Got %s.' % (
              melody_encoder_decoders.keys(), FLAGS.melody_encoder_decoder))
    if None in (FLAGS.generator_id, FLAGS.generator_description):
      raise AccompanimentRnnConfigException(
          '`--generator_id` and `--generator_details` must both be supplied '
          'with `--melody_encoder_decoder`.')
    if FLAGS.generator_id is not None and FLAGS.generator_description:
      generator_details = magenta.protobuf.generator_pb2.GeneratorDetails(
          id=FLAGS.generator_id,
          description=FLAGS.generator_description)
    else:
      generator_details = None
    encoder_decoder = (
        accompaniment_rnn_encoder_decoder.MelodyPairEncoderDecoder(
            melody_encoder_decoders[FLAGS.melody_encoder_decoder],
            FLAGS.predictahead_steps))
    hparams = magenta.common.HParams()
    hparams.parse(FLAGS.hparams)
    return AccompanimentRnnConfig(generator_details, encoder_decoder, hparams)
  else:
    if FLAGS.config not in default_configs:
      raise AccompanimentRnnConfigException(
          '`--config` must be one of %s. Got %s.' % (
              default_configs.keys(), FLAGS.config))
    config = default_configs[FLAGS.config]
    config.hparams.parse(FLAGS.hparams)
    if FLAGS.generator_id is not None:
      config.details.id = FLAGS.generator_id
    if FLAGS.generator_description is not None:
      config.details.description = FLAGS.generator_description
    return config
