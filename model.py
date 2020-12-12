import numpy as np
import tensorflow as tf
import os
from collections import deque
from tqdm import tqdm
import midi

from keras.layers import Input, LSTM, Dense, Dropout, Lambda, Reshape, Permute
from keras.layers import TimeDistributed, RepeatVector, Conv1D, Activation
from keras.layers import Embedding, Flatten
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras import losses

# Define the musical styles
genre = [
    'baroque',
    'classical',
    'romantic'
]
 
styles = [
    [
        #'data/Baroque/bach'
        #'data/Baroque/christmas'
    ],
    [
        'data/Classical/beethoven',
        #'data/Classical/clementi',
        'data/Classical/haydn',
        'data/Classical/mozart'
    ],
    [
        #'data/Romantic/albeniz',
        #'data/Romantic/balakirew',
        #'data/Romantic/borodin',
        #'data/Romantic/brahms',
        #'data/Romantic/burgmueller',
        'data/Romantic/chopin',
        #'data/Romantic/debussy',
        #'data/Romantic/godowsky',
        #'data/Romantic/granados',
        #'data/Romantic/grieg',
        'data/Romantic/liszt',
        #'data/Romantic/mendelssohn',
        #'data/Romantic/moszkowski',
        #'data/Romantic/mussorgsky',
        #'data/Romantic/rachmaninow',
        #'data/Romantic/ravel',
        #'data/Romantic/schubert',
        #'data/Romantic/schumann',
        #'data/Romantic/sinding',
        'data/Romantic/tchaikovsky'
    ]
]
 
NUM_STYLES = sum(len(s) for s in styles)
 
# MIDI Resolution
DEFAULT_RES = 96
MIDI_MAX_NOTES = 128
MAX_VELOCITY = 127
 
# Number of octaves supported
NUM_OCTAVES = 4
OCTAVE = 12
 
# Min and max note (in MIDI note number)
MIN_NOTE = 36
MAX_NOTE = MIN_NOTE + NUM_OCTAVES * OCTAVE #[36-84]
NUM_NOTES = MAX_NOTE - MIN_NOTE
 
# Number of beats in a bar
BEATS_PER_BAR = 4
# Notes per quarter note
NOTES_PER_BEAT = 4
# The quickest note is a half-note
NOTES_PER_BAR = NOTES_PER_BEAT * BEATS_PER_BAR
 
# Training parameters
BATCH_SIZE = 16
SEQ_LEN = 8 * NOTES_PER_BAR
 
# Hyper Parameters
OCTAVE_UNITS = 64
STYLE_UNITS = 64
NOTE_UNITS = 3
TIME_AXIS_UNITS = 256
NOTE_AXIS_UNITS = 128
 
TIME_AXIS_LAYERS = 2
NOTE_AXIS_LAYERS = 2
 
# Move file save location
OUT_DIR = 'out'
MODEL_DIR = os.path.join(OUT_DIR, 'models')
MODEL_FILE = os.path.join(OUT_DIR, 'model.h5')
SAMPLES_DIR = os.path.join(OUT_DIR, 'samples')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')

generation_progress = 0

# Helper functions
def one_hot(i, nb_classes):
    arr = np.zeros((nb_classes,))
    arr[i] = 1
    return arr

def compute_beat(beat, notes_in_bar):
    return one_hot(beat % notes_in_bar, notes_in_bar)
 
def compute_completion(beat, len_melody):
    return np.array([beat / len_melody])
 
def compute_genre(genre_id):
    """ Computes a vector that represents a particular genre """
    genre_hot = np.zeros((NUM_STYLES,))
    start_index = sum(len(s) for i, s in enumerate(styles) if i < genre_id)
    styles_in_genre = len(styles[genre_id])
    genre_hot[start_index:start_index + styles_in_genre] = 1 / styles_in_genre
    return genre_hot
 
def stagger(data, time_steps):
    dataX, dataY = [], []
    # Buffer training for first event
    data = ([np.zeros_like(data[0])] * time_steps) + list(data)
 
    # Chop a sequence into measures
    for i in range(0, len(data) - time_steps, NOTES_PER_BAR):
        dataX.append(data[i:i + time_steps])
        dataY.append(data[i + 1:(i + time_steps + 1)])
    return dataX, dataY

def clamp_midi(sequence):
    """
    Clamps the midi base on the MIN and MAX notes
    """
    return sequence[:, MIN_NOTE:MAX_NOTE, :]
 
def unclamp_midi(sequence):
    """
    Restore clamped MIDI sequence back to MIDI note values
    """
    return np.pad(sequence, ((0, 0), (MIN_NOTE, 0), (0, 0)), 'constant')


# MIDI Util functions
def midi_encode(note_seq, resolution=NOTES_PER_BEAT, step=1):
    """
    Takes a piano roll and encodes it into MIDI pattern
    """
    # Instantiate a MIDI Pattern (contains a list of tracks)
    pattern = midi.Pattern()
    pattern.resolution = resolution
    # Instantiate a MIDI Track (contains a list of MIDI events)
    track = midi.Track()
    # Append the track to the pattern
    pattern.append(track)

    play = note_seq[:, :, 0]
    replay = note_seq[:, :, 1]
    volume = note_seq[:, :, 2]

    # The current pattern being played
    current = np.zeros_like(play[0])
    # Absolute tick of last event
    last_event_tick = 0
    # Amount of NOOP ticks
    noop_ticks = 0

    for tick, data in enumerate(play):
        data = np.array(data)

        if not np.array_equal(current, data):# or np.any(replay[tick]):
            noop_ticks = 0

            for index, next_volume in np.ndenumerate(data):
                if next_volume > 0 and current[index] == 0:
                    # Was off, but now turned on
                    evt = midi.NoteOnEvent(
                        tick=(tick - last_event_tick) * step,
                        velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick
                elif current[index] > 0 and next_volume == 0:
                    # Was on, but now turned off
                    evt = midi.NoteOffEvent(
                        tick=(tick - last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt)
                    last_event_tick = tick

                elif current[index] > 0 and next_volume > 0 and replay[tick][index[0]] > 0:
                    # Handle replay
                    evt_off = midi.NoteOffEvent(
                        tick=(tick- last_event_tick) * step,
                        pitch=index[0]
                    )
                    track.append(evt_off)
                    evt_on = midi.NoteOnEvent(
                        tick=0,
                        velocity=int(volume[tick][index[0]] * MAX_VELOCITY),
                        pitch=index[0]
                    )
                    track.append(evt_on)
                    last_event_tick = tick

        else:
            noop_ticks += 1

        current = data

    tick += 1

    # Turn off all remaining on notes
    for index, vol in np.ndenumerate(current):
        if vol > 0:
            # Was on, but now turned off
            evt = midi.NoteOffEvent(
                tick=(tick - last_event_tick) * step,
                pitch=index[0]
            )
            track.append(evt)
            last_event_tick = tick
            noop_ticks = 0

    # Add the end of track event, append it to the track
    eot = midi.EndOfTrackEvent(tick=noop_ticks)
    track.append(eot)

    return pattern

def midi_decode(pattern,
                classes=MIDI_MAX_NOTES,
                step=None):
    """
    Takes a MIDI pattern and decodes it into a piano roll.
    """
    if step is None:
        step = pattern.resolution // NOTES_PER_BEAT

    # Extract all tracks at highest resolution
    merged_replay = None
    merged_volume = None

    for track in pattern:
        # The downsampled sequences
        replay_sequence = []
        volume_sequence = []

        # Raw sequences
        replay_buffer = [np.zeros((classes,))]
        volume_buffer = [np.zeros((classes,))]

        for i, event in enumerate(track):
            # Duplicate the last note pattern to wait for next event
            for _ in range(event.tick):
                replay_buffer.append(np.zeros(classes))
                volume_buffer.append(np.copy(volume_buffer[-1]))

                # Buffer & downscale sequence
                if len(volume_buffer) > step:
                    # Take the min
                    replay_any = np.minimum(np.sum(replay_buffer[:-1], axis=0), 1)
                    replay_sequence.append(replay_any)

                    # Determine volume by max
                    volume_sum = np.amax(volume_buffer[:-1], axis=0)
                    volume_sequence.append(volume_sum)

                    # Keep the last one (discard things in the middle)
                    replay_buffer = replay_buffer[-1:]
                    volume_buffer = volume_buffer[-1:]

            if isinstance(event, midi.EndOfTrackEvent):
                break

            # Modify the last note pattern
            if isinstance(event, midi.NoteOnEvent):
                pitch, velocity = event.data
                volume_buffer[-1][pitch] = velocity / MAX_VELOCITY

                # Check for replay_buffer, which is true if the current note was previously played and needs to be replayed
                if len(volume_buffer) > 1 and volume_buffer[-2][pitch] > 0 and volume_buffer[-1][pitch] > 0:
                    replay_buffer[-1][pitch] = 1
                    # Override current volume with previous volume
                    volume_buffer[-1][pitch] = volume_buffer[-2][pitch]

            if isinstance(event, midi.NoteOffEvent):
                pitch, velocity = event.data
                volume_buffer[-1][pitch] = 0

        # Add the remaining
        replay_any = np.minimum(np.sum(replay_buffer, axis=0), 1)
        replay_sequence.append(replay_any)
        volume_sequence.append(volume_buffer[0])

        replay_sequence = np.array(replay_sequence)
        volume_sequence = np.array(volume_sequence)
        assert len(volume_sequence) == len(replay_sequence)

        if merged_volume is None:
            merged_replay = replay_sequence
            merged_volume = volume_sequence
        else:
            # Merge into a single track, padding with zeros of needed
            if len(volume_sequence) > len(merged_volume):
                # Swap variables such that merged_notes is always at least
                # as large as play_sequence
                tmp = replay_sequence
                replay_sequence = merged_replay
                merged_replay = tmp

                tmp = volume_sequence
                volume_sequence = merged_volume
                merged_volume = tmp

            assert len(merged_volume) >= len(volume_sequence)

            diff = len(merged_volume) - len(volume_sequence)
            merged_replay += np.pad(replay_sequence, ((0, diff), (0, 0)), 'constant')
            merged_volume += np.pad(volume_sequence, ((0, diff), (0, 0)), 'constant')

    merged = np.stack([np.ceil(merged_volume), merged_replay, merged_volume], axis=2)
    # Prevent stacking duplicate notes to exceed one.
    merged = np.minimum(merged, 1)
    return merged


# Model functions
def primary_loss(y_true, y_pred):
    # 3 separate loss calculations based on if note is played or not
    played = y_true[:, :, :, 0]
    bce_note = losses.binary_crossentropy(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    bce_replay = losses.binary_crossentropy(y_true[:, :, :, 1], tf.multiply(played, y_pred[:, :, :, 1]) + tf.multiply(1 - played, y_true[:, :, :, 1]))
    mse = losses.mean_squared_error(y_true[:, :, :, 2], tf.multiply(played, y_pred[:, :, :, 2]) + tf.multiply(1 - played, y_true[:, :, :, 2]))
    return bce_note + bce_replay + mse
 
def pitch_pos_in_f(time_steps):
    """
    Returns a constant containing pitch position of each note
    """
    def f(x):
        note_ranges = tf.range(NUM_NOTES, dtype='float32') / NUM_NOTES
        repeated_ranges = tf.tile(note_ranges, [tf.shape(x)[0] * time_steps])
        return tf.reshape(repeated_ranges, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
    return f
 
def pitch_class_in_f(time_steps):
    """
    Returns a constant containing pitch class of each note
    """
    def f(x):
        pitch_class_matrix = np.array([one_hot(n % OCTAVE, OCTAVE) for n in range(NUM_NOTES)])
        pitch_class_matrix = tf.constant(pitch_class_matrix, dtype='float32')
        pitch_class_matrix = tf.reshape(pitch_class_matrix, [1, 1, NUM_NOTES, OCTAVE])
        return tf.tile(pitch_class_matrix, [tf.shape(x)[0], time_steps, 1, 1])
    return f
 
def pitch_bins_f(time_steps):
    def f(x):
        bins = tf.reduce_sum([x[:, :, i::OCTAVE, 0] for i in range(OCTAVE)], axis=3)
        bins = tf.tile(bins, [NUM_OCTAVES, 1, 1])
        bins = tf.reshape(bins, [tf.shape(x)[0], time_steps, NUM_NOTES, 1])
        return bins
    return f
 
def time_axis(dropout):
    def f(notes, beat, style):
        time_steps = int(notes.get_shape()[1])
 
        # TODO: Experiment with when to apply conv
        note_octave = TimeDistributed(Conv1D(OCTAVE_UNITS, 2 * OCTAVE, padding='same'))(notes)
        note_octave = Activation('tanh')(note_octave)
        note_octave = Dropout(dropout)(note_octave)
 
        # Create features for every single note.
        note_features = Concatenate()([
            Lambda(pitch_pos_in_f(time_steps))(notes),
            Lambda(pitch_class_in_f(time_steps))(notes),
            Lambda(pitch_bins_f(time_steps))(notes),
            note_octave,
            TimeDistributed(RepeatVector(NUM_NOTES))(beat)
        ])
 
        x = note_features
 
        # [batch, notes, time, features]
        x = Permute((2, 1, 3))(x)
 
        # Apply LSTMs
        for l in range(TIME_AXIS_LAYERS):
            # Integrate style
            style_proj = Dense(int(x.get_shape()[3]))(style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Activation('tanh')(style_proj)
            style_proj = Dropout(dropout)(style_proj)
            style_proj = Permute((2, 1, 3))(style_proj)
            x = Add()([x, style_proj])
 
            x = TimeDistributed(LSTM(TIME_AXIS_UNITS, return_sequences=True))(x)
            x = Dropout(dropout)(x)
 
        # [batch, time, notes, features]
        return Permute((2, 1, 3))(x)
    return f
 
def note_axis(dropout):
    dense_layer_cache = {}
    lstm_layer_cache = {}
    note_dense = Dense(2, activation='sigmoid', name='note_dense')
    volume_dense = Dense(1, name='volume_dense')
 
    def f(x, chosen, style):
        time_steps = int(x.get_shape()[1])
 
        # Shift target one note to the left.
        shift_chosen = Lambda(lambda x: tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]]))(chosen)
 
        # [batch, time, notes, 1]
        shift_chosen = Reshape((time_steps, NUM_NOTES, -1))(shift_chosen)
        # [batch, time, notes, features + 1]
        x = Concatenate(axis=3)([x, shift_chosen])
 
        for l in range(NOTE_AXIS_LAYERS):
            # Integrate style
            if l not in dense_layer_cache:
                dense_layer_cache[l] = Dense(int(x.get_shape()[3]))
 
            style_proj = dense_layer_cache[l](style)
            style_proj = TimeDistributed(RepeatVector(NUM_NOTES))(style_proj)
            style_proj = Activation('tanh')(style_proj)
            style_proj = Dropout(dropout)(style_proj)
            x = Add()([x, style_proj])
 
            if l not in lstm_layer_cache:
                lstm_layer_cache[l] = LSTM(NOTE_AXIS_UNITS, return_sequences=True)
 
            x = TimeDistributed(lstm_layer_cache[l])(x)
            x = Dropout(dropout)(x)
 
        return Concatenate()([note_dense(x), volume_dense(x)])
    return f

def build_models(time_steps=SEQ_LEN, input_dropout=0.2, dropout=0.5):
    notes_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
    beat_in = Input((time_steps, NOTES_PER_BAR))
    style_in = Input((time_steps, NUM_STYLES))
    # Target input for conditioning
    chosen_in = Input((time_steps, NUM_NOTES, NOTE_UNITS))
 
    # Dropout inputs
    notes = Dropout(input_dropout)(notes_in)
    beat = Dropout(input_dropout)(beat_in)
    chosen = Dropout(input_dropout)(chosen_in)
 
    # Distributed representations
    style_l = Dense(STYLE_UNITS, name='style')
    style = style_l(style_in)
 
    """ Time axis """
    time_out = time_axis(dropout)(notes, beat, style)
 
    """ Note Axis & Prediction Layer """
    naxis = note_axis(dropout)
    notes_out = naxis(time_out, chosen, style)
 
    model = Model([notes_in, chosen_in, beat_in, style_in], [notes_out])
    model.compile(optimizer='nadam', loss=[primary_loss])
 
    """ Generation Models """
    time_model = Model([notes_in, beat_in, style_in], [time_out])
 
    note_features = Input((1, NUM_NOTES, TIME_AXIS_UNITS), name='note_features')
    chosen_gen_in = Input((1, NUM_NOTES, NOTE_UNITS), name='chosen_gen_in')
    style_gen_in = Input((1, NUM_STYLES), name='style_in')
 
    # Dropout inputs
    chosen_gen = Dropout(input_dropout)(chosen_gen_in)
    style_gen = style_l(style_gen_in)
 
    note_gen_out = naxis(note_features, chosen_gen, style_gen)
 
    note_model = Model([note_features, chosen_gen_in, style_gen_in], note_gen_out)
 
    return model, time_model, note_model

# Generation functions
class MusicGeneration:
    """
    Represents a music generation
    """
    def __init__(self, style, sequence=None, default_temp=1):
        
        if sequence is None:
            self.notes_memory = deque([np.zeros((NUM_NOTES, NOTE_UNITS)) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
            self.beat_memory = deque([np.zeros(NOTES_PER_BAR) for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
            self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        else:
       
            #self.notes_memory = list(sequence[:128, :, :])
            #self.beat_memory = [list(compute_beat(i, NOTES_PER_BAR)) for i in range(SEQ_LEN)]
            self.notes_memory = deque([sequence[0, :, :] for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
            self.beat_memory = deque([compute_beat(i, NOTES_PER_BAR) for i in range(SEQ_LEN)], maxlen=SEQ_LEN)
            self.style_memory = deque([style for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
 
        # The next note being built
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        self.silent_time = NOTES_PER_BAR
 
        # The outputs
        self.results = []
        # The temperature
        self.default_temp = default_temp
        self.temperature = default_temp
 
    def build_time_inputs(self):
        return (
            np.array(self.notes_memory),
            np.array(self.beat_memory),
            np.array(self.style_memory)
        )
 
    def build_note_inputs(self, note_features):
        # Timesteps = 1 (No temporal dimension)
        return (
            np.array(note_features),
            np.array([self.next_note]),
            np.array(list(self.style_memory)[-1:])
        )
 
    def choose(self, prob, n):
        vol = prob[n, -1]
        prob = apply_temperature(prob[n, :-1], self.temperature)
 
        # Flip notes randomly
        if np.random.random() <= prob[0]:
            self.next_note[n, 0] = 1
            # Apply volume
            self.next_note[n, 2] = vol
            # Flip articulation
            if np.random.random() <= prob[1]:
                self.next_note[n, 1] = 1
 
    def end_time(self, t):
        """
        Finish generation for this time step.
        """
        # Increase temperature while silent.
        if np.count_nonzero(self.next_note) == 0:
            self.silent_time += 1
            if self.silent_time >= NOTES_PER_BAR:
                self.temperature += 0.1
        else:
            self.silent_time = 0
            self.temperature = self.default_temp
 
        self.notes_memory.append(self.next_note)
        # Consistent with dataset representation
        self.beat_memory.append(compute_beat(t, NOTES_PER_BAR))
        self.results.append(self.next_note)
        # Reset next note
        self.next_note = np.zeros((NUM_NOTES, NOTE_UNITS))
        return self.results[-1]
    
def apply_temperature(prob, temperature):
    """
    Applies temperature to a sigmoid vector.
    """
    # Apply temperature
    if temperature != 1:
        # Inverse sigmoid
        x = -np.log(1 / prob - 1)
        # Apply temperature to sigmoid function
        prob = 1 / (1 + np.exp(-x / temperature))
    return prob

def process_inputs(ins):
    ins = list(zip(*ins))
    ins = [np.array(i) for i in ins]
    return ins

def generate(models, num_bars, styles):
    print('Generating with styles:', styles)

    _, time_model, note_model = models
    generations = [MusicGeneration(style, sequence=None) for style in styles]
        
    for t in tqdm(range(NOTES_PER_BAR * num_bars)): 
        #generation_progress =  int(np.round(t/(NOTES_PER_BAR * num_bars)*100, 0))
        #if t%5==0 or t==(NOTES_PER_BAR * num_bars)-1:
            #open('generation_progress.txt', 'w').write(str(generation_progress))
        #print(f"Generation progress: {generation_progress}%", end='\r')
        # Produce note-invariant features
        ins = process_inputs([g.build_time_inputs() for g in generations])
        #ins = [[g.build_time_inputs() for g in generations]]
        # Pick only the last time step
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([g.build_note_inputs(note_features[i, :, :, :]) for i, g in enumerate(generations)])
            predictions = np.array(note_model.predict(ins))

            for i, g in enumerate(generations):
                # Remove the temporal dimension
                g.choose(predictions[i][-1], n)

        # Move one time step
        yield [g.end_time(t) for g in generations]

def generate_with_seq(models, num_bars, style, sequence):
    print('Generating with styles:', styles)

    _, time_model, note_model = models
    generation = MusicGeneration(style, sequence)
    
    for t in tqdm(range(NOTES_PER_BAR * num_bars)):
        # Produce note-invariant features
        #generation_progress =  int(np.round(t/(NOTES_PER_BAR * num_bars)*100, 0))
        #if t%5==0 or t==(NOTES_PER_BAR * num_bars)-1:
            #open('generation_progress.txt', 'w').write(str(generation_progress))
        #print(f"Generation progress: {generation_progress}%", end='\r')
        ins = process_inputs([generation.build_time_inputs()])
        #ins = [[g.build_time_inputs() for g in generations]]
        # Pick only the last time step
        note_features = time_model.predict(ins)
        note_features = np.array(note_features)[:, -1:, :]

        # Generate each note conditioned on previous
        for n in range(NUM_NOTES):
            ins = process_inputs([generation.build_note_inputs(note_features[0, :, :, :])])
            predictions = np.array(note_model.predict(ins))

            generation.choose(predictions[0][-1], n)

        # Move one time step
        yield [generation.end_time(t)]
