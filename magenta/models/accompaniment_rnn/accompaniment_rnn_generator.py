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
"""Accompaniment RNN generation code as a SequenceGenerator interface."""

import copy
from functools import partial
import random

# internal imports
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf

import magenta

from magenta.models.accompaniment_rnn import accompaniment_rnn_config
from magenta.models.accompaniment_rnn import accompaniment_rnn_encoder_decoder
from magenta.models.accompaniment_rnn import accompaniment_rnn_graph


STEPS_PER_QUARTER = 4


class AccompanimentRnnSequenceGenerator(magenta.music.BaseSequenceGenerator):
  """Accompaniment RNN generation code as a SequenceGenerator interface.

  Generation is applied to instrument 1 of the NoteSequence conditioned on
  instrument 0.

  Args:
      config: An AccompanimentRnnConfig containing the GeneratorDetails,
          MelodyEncoderDecoder, and HParams to use.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
  """

  def __init__(self, config, checkpoint=None, bundle=None):
    super(AccompanimentRnnSequenceGenerator, self).__init__(
        config.details, checkpoint, bundle)
    self._session = None

    self._steps_per_quarter = STEPS_PER_QUARTER
    self._config = config
    # Override hparams for generation.
    self._config.hparams.dropout_keep_prob = 1.0
    self._config.hparams.batch_size = 1

  @property
  def predictahead_steps(self):
    return self._config.predictahead_steps

  def _initialize_with_checkpoint(self, checkpoint_file):
    graph = accompaniment_rnn_graph.build_graph('generate', self._config)
    with graph.as_default():
      saver = tf.train.Saver()
      self._session = tf.Session()
      tf.logging.info('Checkpoint used: %s', checkpoint_file)
      saver.restore(self._session, checkpoint_file)

  def _initialize_with_checkpoint_and_metagraph(self, checkpoint_filename,
                                                metagraph_filename):
    with tf.Graph().as_default():
      self._session = tf.Session()
      new_saver = tf.train.import_meta_graph(metagraph_filename)
      new_saver.restore(self._session, checkpoint_filename)

  def _write_checkpoint_with_metagraph(self, checkpoint_filename):
    with self._session.graph.as_default():
      saver = tf.train.Saver(sharded=False)
      saver.save(self._session, checkpoint_filename, meta_graph_suffix='meta',
                 write_meta_graph=True)

  def _close(self):
    self._session.close()
    self._session = None

  def _seconds_to_steps(self, seconds, qpm):
    """Converts seconds to steps.

    Uses the generator's _steps_per_quarter setting and the specified qpm.

    Args:
      seconds: number of seconds.
      qpm: current qpm.

    Returns:
      Number of steps the seconds represent.
    """
    return int(seconds * (qpm / 60.0) * self._steps_per_quarter)

  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
      raise magenta.music.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise magenta.music.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    temperature = (generator_options.args['temperature'].float_value
                   if 'temperature' in generator_options.args else 1.0)
    generate_section = generator_options.generate_sections[0]

    # If input section exists, use it to limit the input sequence.
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      input_sequence = magenta.music.extract_subsequence(
          input_sequence, input_section.start_time, input_section.end_time)

    accompaniment_end_times = [
        n.end_time for n in input_sequence.notes if n.instrument == 1]
    accompaniment_end_time = (max(accompaniment_end_times)
                              if accompaniment_end_times else 0)
    if accompaniment_end_time > generate_section.start_time:
      raise magenta.music.SequenceGeneratorException(
          'Got GenerateSection request for section that is before the end of '
          'the accompaniment. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time, accompaniment_end_time))

    # Quantize the priming sequence.
    quantized_sequence = magenta.music.QuantizedSequence()
    quantized_sequence.from_note_sequence(input_sequence,
                                          self._steps_per_quarter)
    # Setting gap_bars to infinite ensures that the entire input will be used.
    extracted_melodies, _ = magenta.music.extract_melodies(
        quantized_sequence,
        min_bars=0,
        min_unique_pitches=1,
        gap_bars=float('inf'),
        ignore_polyphonic_notes=True)
    assert 1 <= len(extracted_melodies) <= 2

    qpm = (input_sequence.tempos[0].qpm if input_sequence and
           input_sequence.tempos else magenta.music.DEFAULT_QUARTERS_PER_MINUTE)
    start_step = self._seconds_to_steps(generate_section.start_time, qpm)
    end_step = self._seconds_to_steps(generate_section.end_time, qpm)

    encoder_decoder = self._config.encoder_decoder
    predictahead_steps = encoder_decoder.predictahead_steps
    if start_step < predictahead_steps:
      raise magenta.music.SequenceGeneratorException(
          'Got GenerateSection request for section that is before the '
          'earliest possible prediction time. This model can only extend '
          'sequences after %d steps, but step %d was requested.' %
          (predictahead_steps, start_step))

    if len(extracted_melodies) == 2:
      main_melody, accompaniment = extracted_melodies
    elif len(extracted_melodies) == 1:
      # TODO(adarob): This might actually be the accompaniment.
      main_melody = extracted_melodies[0]
      tf.logging.warn(
          'No accompaniment was extracted from the priming sequence. '
          'Priming will be generated from scratch.')
      accompaniment = magenta.music.Melody(
          [magenta.music.constants.MELODY_NO_EVENT] *
          (start_step - main_melody.start_step) +
          [random.randint(16, 54)],
          start_step=main_melody.start_step)
      start_step += 1
    else:
      raise magenta.music.SequenceGeneratorException(
          'Input sequence should have between 1 and 2 extractable melodies. '
          'Got %d.' % len(extracted_melodies))

    transpose_amounts = (
        main_melody.squash(encoder_decoder.min_note,
                           encoder_decoder.max_note,
                           encoder_decoder.transpose_to_key),
        accompaniment.squash(encoder_decoder.min_note,
                             encoder_decoder.max_note,
                             encoder_decoder.transpose_to_key))

    # Ensure that the accompaniment extends up to the step we want to start
    # generating.
    accompaniment.set_length(start_step - accompaniment.start_step)

    graph_inputs = self._session.graph.get_collection('inputs')[0]
    graph_initial_state = self._session.graph.get_collection('initial_state')[0]
    graph_final_state = self._session.graph.get_collection('final_state')[0]
    graph_softmax = self._session.graph.get_collection('softmax')[0]
    graph_temperature = self._session.graph.get_collection('temperature')[0]

    final_state = None
    for i in range(end_step - accompaniment.end_step):
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
    sequence = main_melody.to_sequence(instrument=0, qpm=qpm)
    sequence.notes.extend(
        accompaniment.to_sequence(instrument=1, qpm=qpm).notes)
    return sequence


def get_generator_map():
  """Returns a map from the generator ID to its SequenceGenerator class.

  Binds the `config` argument so that the constructor matches the
  BaseSequenceGenerator class.

  Returns:
    Map from the generator ID to its SequenceGenerator class with a bound
    `config` argument.
  """
  return {key: partial(AccompanimentRnnSequenceGenerator, config)
          for (key, config) in accompaniment_rnn_config.default_configs.items()}
