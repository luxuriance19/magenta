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
"""A class for encoding/decoding pairs of melodies to predict accompaniments."""

import collections
import copy

#internal imports

import magenta

# A pair of Melody objects.
MelodyPair = collections.namedtuple('MelodyPair', ['given', 'predicted'])


class AccompanimentRnnEncoderDecoder(object):
  """An encoder/decoder specific to the Accompaniment RNN model.

  Encodes a MelodyPair for prediction of one melody conditioned on the other.
  Optionally predicts some number of steps in the future (`predictahead_steps`).

  Uses a MelodyEncoderDecoder to encode the individual melodies, concatenating
  the inputs and using only the predicted melody as output, after offsetting
  the two melodies appropriately.

  Attributes:
    individual_melody_encoder_decoder: A MelodyEncoderDecoder object to use for
        encoding/decoding the individual melodies.
    predictahead_steps: The number of steps ahead of the conditioned melody to
        output the predicted melody. In other words, the output for step t
        will be conditioned on the given melody up to step
        t-`predictahead_steps` and the predicted melody from step
        `predictahead_steps` to step t-1.

    Raises:
      ValueError: If `predicathead_steps` is negative.
  """

  def __init__(self,
               individual_melody_encoder_decoder,
               predictahead_steps=0):
    """Initializes the MelodyEncoderDecoder."""
    self._melody_encoder_decoder = individual_melody_encoder_decoder

    if predictahead_steps < 0:
      raise ValueError('`predictahead_steps` must be non-negative. Got: %d.' %
                       predictahead_steps)
    self._predictahead_steps = predictahead_steps

  @property
  def min_note(self):
    return self._melody_encoder_decoder.min_note

  @property
  def max_note(self):
    return self._melody_encoder_decoder.max_note

  @property
  def transpose_to_key(self):
    return self._melody_encoder_decoder.transpose_to_key

  @property
  def input_size(self):
    return self._melody_encoder_decoder.input_size * 2

  @property
  def num_classes(self):
    return self._melody_encoder_decoder.num_classes

  @property
  def no_event_label(self):
    return self._melody_encoder_decoder.no_event_label

  @property
  def predictahead_steps(self):
    return self._predictahead_steps

  def _encode_individual_melody_inputs(self, melody):
    inputs = []
    for i in range(len(melody)):
      inputs.append(self._melody_encoder_decoder.events_to_input(melody, i))
    return inputs

  def _get_inputs(self, given_melody, predicted_melody, full_length):
    """Assumes squashed melodies."""
    assert len(given_melody) == len(predicted_melody)

    # Make deep copies so as to not modify the originals.
    given_melody = copy.deepcopy(given_melody)
    predicted_melody = copy.deepcopy(predicted_melody)

    initial_length = len(given_melody)

    if self._predictahead_steps == 0:
      final_length = initial_length
      # Add padding to the left.
      predicted_melody.set_length(initial_length + 1, from_left=True)
      # Remove overhang from right.
      predicted_melody.set_length(final_length, from_left=False)
    else:
      final_length = initial_length - self._predictahead_steps + 1
      # Remove overhang from right.
      given_melody.set_length(final_length, from_left=False)
      # Remove predictahead_steps from the left.
      predicted_melody.set_length(final_length - 1, from_left=True)
      # Add single space of padding to the left.
      predicted_melody.set_length(final_length, from_left=True)

    assert len(given_melody) == len(predicted_melody)

    # Concatenate the encoded inputs for the two melodies at each step in the
    # full length case and at the final step otherwise.
    if full_length:
      return [individual_inputs[0] + individual_inputs[1]
              for individual_inputs in zip(
                  self._encode_individual_melody_inputs(given_melody),
                  self._encode_individual_melody_inputs(predicted_melody))]
    else:
      return [self._melody_encoder_decoder.events_to_input(given_melody,
                                                           final_length - 1) +
              self._melody_encoder_decoder.events_to_input(predicted_melody,
                                                           final_length - 1)]

  def encode(self, melody_pair):
    """Returns a SequenceExample for the given MelodyPair.

    Args:
      melody_pair: A MelodyPair object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    given_melody = melody_pair.given
    predicted_melody = melody_pair.predicted

    inputs = self._get_inputs(given_melody, predicted_melody, full_length=True)
    if self._predictahead_steps > 0:
      # We won't have an output value for the final step.
      del inputs[-1]

    labels = []
    start_position = self._predictahead_steps

    for i in range(start_position, len(predicted_melody)):
      labels.append(
          self._melody_encoder_decoder.events_to_label(predicted_melody, i))

    assert len(inputs) == len(labels)
    return magenta.common.make_sequence_example(inputs, labels)

  def squash_and_encode(self, melody_pair):
    """Returns a SequenceExample for the given melody pair after squashing.

    Args:
      melody_pair: A MelodyPair object.

    Returns:
      A tf.train.SequenceExample containing inputs and labels.
    """
    melody_pair[0].squash(self.min_note, self.max_note, self.transpose_to_key)
    melody_pair[1].squash(self.min_note, self.max_note, self.transpose_to_key)
    return self.encode(melody_pair)

  def get_inputs_batch(self, melody_pairs, full_length=False):
    """Returns an inputs batch for the given melodies.

    Args:
      melody_pairs: A list of MelodyPair objects.
      full_length: If True, the inputs batch will be for the full length of
          each melody. If False, the inputs batch will only be for the last
          event of each melody. A full-length inputs batch is used for the
          first step of extending the melodies, since the rnn cell state needs
          to be initialized with the priming pair. For subsequent generation
          steps, only a last-event inputs batch is used.

    Returns:
      An inputs batch. If `full_length` is True, the shape will be
      [len(melody_pairs), len(melody_pairs[0]), INPUT_SIZE]. If `full_length`
      is False, the shape will be [len(melody_pairs), 1, INPUT_SIZE].
    """
    inputs_batch = []
    for melody_pair in melody_pairs:
      inputs_batch.append(
          self._get_inputs(melody_pair.given, melody_pair.predicted,
                           full_length))
    return inputs_batch

  def extend_melodies(self, melodies, softmax):
    self._melody_encoder_decoder.extend_event_sequences(melodies, softmax)
