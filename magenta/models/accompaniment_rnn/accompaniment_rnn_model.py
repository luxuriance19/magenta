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
"""Accompaniment RNN model."""

import copy

# internal imports

from six.moves import range  # pylint: disable=redefined-builtin

import magenta
from magenta.models.accompaniment_rnn import accompaniment_rnn_encoder_decoder
from magenta.models.accompaniment_rnn import accompaniment_rnn_graph


class AccompanimentRnnModelException(Exception):
  pass


class AccompanimentRnnModel(magenta.music.BaseModel):
  """Class for accompaniment RNN generation models.

  Currently this class only supports generation, of both MelodyPairs and
  NoteSequences (containing MelodyPairs). Support for model training will be
  added at a later time.
  """

  def __init__(self, config):
    """Initialize the AccompanimentRnnConfig.

    Args:
      config: A AccompanimentRnnConfig.
    """
    super(AccompanimentRnnModel, self).__init__()
    self._config = config

    # Override hparams for generation.
    # TODO(fjord): once this class supports training, make this step conditional
    # on the usage mode.
    self._config.hparams.dropout_keep_prob = 1.0
    self._config.hparams.batch_size = 1

  def _build_graph_for_generation(self):
    return accompaniment_rnn_graph.build_graph('generate', self._config)

  @property
  def predictahead_steps(self):
    return self._config.encoder_decoder.predictahead_steps

  def generate_melody_pair(self, num_steps, primer_melody_pair,
                           temperature=1.0):
    """Generate a MelodyPair from a primer melody.

    Args:
      num_steps: The integer length in steps of the final melody pair, after
          generation. Includes the primer.
      primer_melody_pair: The primer melody pair, a MelodyPair object.
      temperature: A float specifying how much to divide the logits by
         before computing the softmax. Greater than 1.0 makes melodies more
         random, less than 1.0 makes melodies less random.

    Returns:
      The generated MelodyPair object (which includes with the provided primer
          melody pair).
    """
    main_melody, accompaniment = copy.deepcopy(primer_melody_pair)
    encoder_decoder = self._config.encoder_decoder
    transpose_amounts = (
        main_melody.squash(encoder_decoder.min_note,
                           encoder_decoder.max_note,
                           encoder_decoder.transpose_to_key),
        accompaniment.squash(encoder_decoder.min_note,
                             encoder_decoder.max_note,
                             encoder_decoder.transpose_to_key))

    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_initial_state = self._session.graph.get_collection('initial_state')[0]
    graph_final_state = self._session.graph.get_collection('final_state')[0]
    graph_softmax = self._session.graph.get_collection('softmax')[0]
    graph_temperature = self._session.graph.get_collection('temperature')[0]

    final_state = None
    for i in range(num_steps):
      if encoder_decoder.predictahead_steps == 0:
        # Add padding.
        accompaniment.set_length(len(accompaniment) + 1)

      main_melody_prefix = copy.deepcopy(main_melody)
      main_melody_prefix.set_length(len(accompaniment))
      melody_pair = accompaniment_rnn_encoder_decoder.MelodyPair(
          main_melody_prefix, accompaniment)
      if i == 0:
        inputs = encoder_decoder.get_inputs_batch([melody_pair],
                                                  full_length=True)
        initial_state = self._session.run(graph_initial_state)
      else:
        inputs = encoder_decoder.get_inputs_batch([melody_pair])
        initial_state = final_state

      feed_dict = {graph_inputs: inputs,
                   graph_initial_state: initial_state,
                   graph_temperature: temperature}
      final_state, softmax = self._session.run(
          [graph_final_state, graph_softmax], feed_dict)
      if encoder_decoder.predictahead_steps == 0:
        # Remove padding.
        accompaniment.set_length(len(accompaniment) - 1)

      encoder_decoder.extend_melodies([accompaniment], softmax)

    main_melody.transpose(-transpose_amounts[0])
    accompaniment.transpose(-transpose_amounts[1])

    return accompaniment_rnn_encoder_decoder.MelodyPair(
        main_melody, accompaniment)


class AccompanimentRnnConfig(object):
  """Stores a configuration for an AccompanimentRnnModel.

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
    'accompaniment_att40_512x3_p2': AccompanimentRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='accompaniment_att40_512x3_p2',
            description='Accompaniment RNN with lookback encoding, 3 512-node '
                        'layers, 40-length attention, and 2-step lookahead.'),
        accompaniment_rnn_encoder_decoder.AccompanimentRnnEncoderDecoder(
            magenta.music.LookbackMelodyEncoderDecoder(),
            predictahead_steps=2),
        magenta.common.HParams(
            batch_size=128,
            rnn_layer_sizes=[512, 512, 512],
            dropout_keep_prob=0.5,
            skip_first_n_losses=0,
            attn_length=40,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97)),
    'accompaniment_att40_512x3_p2_notranspose': AccompanimentRnnConfig(
        magenta.protobuf.generator_pb2.GeneratorDetails(
            id='accompaniment_att40_512x3_p2_notranspose',
            description='Accompaniment RNN with lookback encoding, 3 512-node '
                        'layers, 40-length attention, 2-step lookahead, and '
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
            attn_length=40,
            clip_norm=3,
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.97))
}
