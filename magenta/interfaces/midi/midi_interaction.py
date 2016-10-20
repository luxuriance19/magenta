"""A module for implementing interaction between MIDI and SequenceGenerators."""

import abc
import threading
import time

# internal imports
import magenta
import tensorflow as tf

from magenta.common import concurrency
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2


class MidiInteractionException(Exception):
  """Base class for exceptions in this module."""
  pass


# TODO(adarob): Move to sequence_utils.
def merge_sequence_notes(sequence_1, sequence_2):
  """Returns a new NoteSequence combining the notes from both inputs.

  All fields aside from `notes` and `total_time` are copied from the first
  input.

  Args:
    sequence_1: A NoteSequence to merge. All fields aside from `notes` and
        `total_time` are copied directly from this sequence in the merged
        sequence.
    sequence_2: A NoteSequence to merge.

  Returns:
    A new NoteSequence combining the notes from the input sequences.
  """
  merged_sequence = music_pb2.NoteSequence()
  merged_sequence.CopyFrom(sequence_1)
  merged_sequence.notes.extend(sequence_2.notes)
  merged_sequence.total_time = max(sequence_1.total_time, sequence_2.total_time)
  return merged_sequence


# TODO(adarob): Move to sequence_utils.
def filter_instrument(sequence, instrument, from_time=0):
  """Returns a new NoteSequence with notes from the given instrument removed.

  Only notes that start on or after `from_time` will be completely removed.
  Those that start before and end after `from_time` will be truncated to end
  at `from_time`.

  Args:
    sequence: The NoteSequence to created the filtered sequence from.
    instrument: The instrument number to remove notes of.
    from_time: The time on or after which to remove or truncate notes.

  Returns:
    A new NoteSequence with notes from the given instrument removed or truncated
    after `from_time`.
  """
  filtered_sequence = music_pb2.NoteSequence()
  filtered_sequence.CopyFrom(sequence)
  del filtered_sequence.notes[:]
  for note in sequence.notes:
    if note.instrument == instrument:
      if note.start_time >= from_time:
        continue
      if note.end_time >= from_time:
        note.end_time = from_time
    filtered_sequence.notes.add().CopyFrom(note)
  return filtered_sequence


class MidiInteraction(threading.Thread):
  """Base class for handling interaction between MIDI and SequenceGenerator.

  Child classes will provided the "main loop" of an interactive session between
  a MidiHub used for MIDI I/O and sequences generated by a SequenceGenerator in
  their `run` methods.

  Should be started by calling `start` to launch in a separate thread.

  Args:
    midi_hub: The MidiHub to use for MIDI I/O.
    qpm: The quarters per minute to use for this interaction.
  """
  _metaclass__ = abc.ABCMeta

  def __init__(self, midi_hub, qpm):
    self._midi_hub = midi_hub
    self._qpm = qpm
    # A signal to tell the main loop when to stop.
    self._stop_signal = threading.Event()
    super(MidiInteraction, self).__init__()

  @abc.abstractmethod
  def run(self):
    """The main loop for the interaction.

    Must exit shortly after `self._stop_signal` is set.
    """
    pass

  def stop(self):
    """Stops the main loop, and blocks until the interaction is stopped."""
    self._stop_signal.set()
    self.join()


class CallAndResponseMidiInteraction(MidiInteraction):
  """Implementation of a MidiInteraction for real-time "call and response".

  Alternates between receiving input from the MidiHub ("call") and playing
  generated sequences ("response"). During the call stage, the input is captured
  and used to generate the response, which is then played back during the
  response stage.

  Args:
    midi_hub: The MidiHub to use for MIDI I/O.
    qpm: The quarters per minute to use for this interaction.
    sequence_generator: The SequenceGenerator to use to generate the responses
        in this interaction.
    quarters_per_bar: The number of quarter notes in each bar/measure.
    phrase_bars: The optional number of bars in each phrase. `end_call_signal`
        must be provided if None.
    end_call_signal: The optional midi_hub.MidiSignal to use as a signal to stop
        the call phrase at the end of the current bar. `phrase_bars` must be
        provided if None.
  """
  _MIN_PREDICTAHEAD_QUARTERS = 1

  def __init__(self,
               midi_hub,
               qpm,
               sequence_generator,
               quarters_per_bar=4,
               phrase_bars=None,
               end_call_signal=None):
    super(CallAndResponseMidiInteraction, self).__init__(midi_hub, qpm)
    self._sequence_generator = sequence_generator
    self._quarters_per_bar = quarters_per_bar
    self._phrase_bars = phrase_bars
    self._end_call_signal = end_call_signal

  def run(self):
    """The main loop for a real-time call and response interaction."""

    # We measure time in units of quarter notes.
    quarter_duration = 60.0 / self._qpm
    # Start time in quarter notes from the epoch.
    start_quarters = (time.time() + 1.0) // quarter_duration

    # The number of notes before call stage ends to start generation of
    # response. Will be automatically adjusted to be as small as possible while
    # avoiding late response starts.
    predictahead_quarters = self._MIN_PREDICTAHEAD_QUARTERS

    # Call stage start in quarter notes from the epoch.
    call_start_quarters = start_quarters

    while not self._stop_signal.is_set():
      # Call stage.

      # Start the metronome at the beginning of the call stage.
      self._midi_hub.start_metronome(
          self._qpm, call_start_quarters * quarter_duration)

      # Start a captor at the beginning of the call stage.
      captor = self._midi_hub.start_capture(
          self._qpm, call_start_quarters * quarter_duration)

      if self._phrase_bars is not None:
        # The duration of the call stage in quarter notes.
        call_quarters = self._phrase_bars * self._quarters_per_bar
      else:
        # Wait for end signal.
        self._midi_hub.wait_for_event(self._end_call_signal)
        # The duration of the call stage in quarter notes.
        # We end the call stage at the end of the next bar that is at least
        # `predicathead_quarters` in the future.
        call_quarters = time.time() // quarter_duration - call_start_quarters
        remaining_call_quarters = -call_quarters % self._quarters_per_bar
        if remaining_call_quarters < predictahead_quarters:
          remaining_call_quarters += self._quarters_per_bar
        call_quarters += remaining_call_quarters

      # Set the metronome to stop at the appropriate time.
      self._midi_hub.stop_metronome(
          (call_quarters + call_start_quarters) * quarter_duration,
          block=False)

      # Stop the captor at the appropriate time.
      capture_quarters = call_quarters - predictahead_quarters
      captor.stop(stop_time=(
          (capture_quarters + call_start_quarters) * quarter_duration))
      captured_sequence = captor.captured_sequence()

      # Check to see if a stop has been requested during capture.
      if self._stop_signal.is_set():
        break

      # Generate sequence.
      response_start_quarters = call_quarters + call_start_quarters
      response_end_quarters = 2 * call_quarters + call_start_quarters

      generator_options = generator_pb2.GeneratorOptions()
      generator_options.generate_sections.add(
          start_time=response_start_quarters * quarter_duration,
          end_time=response_end_quarters * quarter_duration)

      # Generate response.
      response_sequence = self._sequence_generator.generate(
          captured_sequence, generator_options)

      # Check to see if a stop has been requested during generation.
      if self._stop_signal.is_set():
        break

      # Response stage.
      # Start response playback.
      self._midi_hub.start_playback(response_sequence)

      # Compute remaining time after generation before the response stage
      # starts, updating `predictahead_quarters` appropriately.
      remaining_time = response_start_quarters * quarter_duration - time.time()
      if remaining_time > (predictahead_quarters * quarter_duration):
        predictahead_quarters = max(self._MIN_PREDICTAHEAD_QUARTERS,
                                    predictahead_quarters - 1)
        tf.logging.info('Generator is ahead by %.3f seconds. '
                        'Decreasing predictahead_quarters to %d.',
                        remaining_time, predictahead_quarters)
      elif remaining_time < 0:
        predictahead_quarters += 1
        tf.logging.info('Generator is lagging by %.3f seconds. '
                        'Increasing predictahead_quarters to %d.',
                        -remaining_time, predictahead_quarters)

      call_start_quarters = response_end_quarters

  def stop(self):
    if self._end_call_signal is not None:
      self._midi_hub.wake_signal_waiters(self._end_call_signal)
    super(CallAndResponseMidiInteraction, self).stop()


class TeacherForcerMidiInteraction(MidiInteraction):
  """Implementation of a MidiInteraction for generating real-time accompaniment.

  Input from the MidiHub is continuously captured and passed to a
  SequenceGenerator to predict what an accompanying voice should play in the
  near future. This generated accompaniment is then played by the MidiHub.

  Args:
    midi_hub: The MidiHub to use for MIDI I/O.
    qpm: The quarters per minute to use for this interaction.
    sequence_generator: The SequenceGenerator to use to generate the
        accompanying voice in this interaction.
    predictahead_quarters: The number of quarter notes to start prediction past
        the end of the captured sequence. Should be determined by how the model
        underlying the generator was trained.
  """
  _MIN_PREDICTAHEAD_STEPS = 1

  def __init__(self, midi_hub, qpm, sequence_generator, phrase_bars,
               steps_per_bar=16, history_steps=48):
    super(TeacherForcerMidiInteraction, self).__init__(midi_hub, qpm)
    self._sequence_generator = sequence_generator
    self._phrase_bars = phrase_bars
    self._steps_per_bar = steps_per_bar
    self._history_steps = history_steps

  def run(self):
    """The main loop for a teacher-forcing interaction.

    Continuously captures input from the MidiHub while repeatedly generating
    additional steps of the accompaniment sequence and playing it back via
    the MidiHub. Stops when `_stop_signal` is set by the `stop` method).
    """
    step_duration = 60.0 / (self._qpm * 4)
    start_steps = (time.time() + 1.0) // step_duration

    # Offset of start of phrase in steps from the epoch.
    phrase_steps = self._phrase_bars * self._steps_per_bar
    phrase_start_steps = start_steps

    # The number of steps before each phrase ends to start generation of next
    # phrase. Will be automatically adjusted to be as small as possible while
    # avoiding late response starts.
    predictahead_steps = self._MIN_PREDICTAHEAD_STEPS

    playback_sequence = music_pb2.NoteSequence()

    # Start metronome.
    self._midi_hub.start_metronome(self._qpm, start_steps * step_duration)
    # Start captor.
    captor = self._midi_hub.start_capture(self._qpm,
                                          start_steps * step_duration)
    # Start player.
    player = self._midi_hub.start_playback(
        playback_sequence, allow_updates=True)
    while not self._stop_signal.is_set():
      if phrase_start_steps > start_steps:
        self._midi_hub.passthrough = False
      # Offset of end of captured sequence in quarter notes from the epoch.
      capture_end_steps = phrase_start_steps + phrase_steps - predictahead_steps

      # TODO(adarob): Move sleep_until inside captor.
      concurrency.Sleeper().sleep_until(capture_end_steps * step_duration)
      captured_sequence = captor.captured_sequence(
          end_time=capture_end_steps * step_duration)

      generation_start_steps = phrase_start_steps + phrase_steps
      generation_end_steps = generation_start_steps + phrase_steps
      generator_options = generator_pb2.GeneratorOptions()
      generator_options.generate_sections.add(
          start_time=generation_start_steps * step_duration,
          end_time=generation_end_steps * step_duration)

      input_sequence = (merge_sequence_notes(
          magenta.music.extract_subsequence(
            playback_sequence,
            (max(start_steps, phrase_start_steps - self._history_steps) *
             step_duration),
            (phrase_start_steps - phrase_steps) * step_duration),
          magenta.music.extract_subsequence(
            captured_sequence,
            (phrase_start_steps - phrase_steps) * step_duration,
            (phrase_start_steps + phrase_steps) * step_duration)))
      # Generate additional accompaniment notes.
      playback_sequence = self._sequence_generator.generate(
          input_sequence, generator_options)

      # Update player with extended accompaniment.
      player.update_sequence(playback_sequence)

      # Update next phrase start.
      phrase_start_steps += phrase_steps

      # Compute remaining time after generation before the next phrase starts,
      # updating `predictahead_steps` appropriately.
      remaining_time = phrase_start_steps * step_duration - time.time()
      if remaining_time > (predictahead_steps * step_duration):
        predictahead_steps = max(self._MIN_PREDICTAHEAD_STEPS,
                                 predictahead_steps - 1)
        tf.logging.info('Generator is ahead by %.3f seconds. '
                        'Decreasing `predictahead_steps` to %d.',
                        remaining_time, predictahead_steps)
      elif remaining_time < 0:
        predictahead_steps += 1
        tf.logging.info('Generator is lagging by %.3f seconds. '
                        'Increasing `predictahead_steps` to %d.',
                        -remaining_time, predictahead_steps)

    # Stop metronome.
    self._midi_hub.stop_metronome()
    # Stop captor.
    captor.stop()
    # Stop player.
    player.stop()
