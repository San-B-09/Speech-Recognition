import numpy as np
from scipy.io import wavfile
#import matplotlib.pyplot as plt
 
DIR = 'C:/Users/bijaw/Desktop/New folder (3)/trial'
fns = ['/bed/00f0204f_nohash_0.wav',
       '/cat/00b01445_nohash_0.wav',
       '/happy/0a2b400e_nohash_0.wav']
SAMPLE_RATE = 16000
 
def read_wav_file(x):
    # Read wavfile using scipy wavfile.read
    _, wav = wavfile.read(x) 
    # Normalize
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
        
    return wav
 
fig = plt.figure(figsize=(14, 8))
for i, fn in enumerate(fns):
    wav = read_wav_file(DIR + fn)
 
    ax = fig.add_subplot(3,1,i+1)
    ax.set_title('Raw wave of ' + fn)
    ax.set_ylabel('Amplitude')
    ax.plot(np.linspace(0, SAMPLE_RATE/len(wav), SAMPLE_RATE), wav)
fig.tight_layout()


from scipy.signal import stft
 
def log_spectrogram(wav):
    freqs, times, spec = stft(wav, SAMPLE_RATE, nperseg = 400, noverlap = 240, nfft = 512, 
                              padded = False, boundary = None)
    # Log spectrogram
    amp = np.log(np.abs(spec)+1e-10)
    
    return freqs, times, amp
 
fig = plt.figure(figsize=(14, 8))
for i, fn in enumerate(fns):
    wav = read_wav_file(DIR + fn)
    freqs, times, amp = log_spectrogram(wav)
    
    ax = fig.add_subplot(3,1,i+1)
    ax.imshow(amp, aspect='auto', origin='lower', 
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    ax.set_title('Spectrogram of ' + fn)
    ax.set_ylabel('Freqs in Hz')
    ax.set_xlabel('Seconds')
fig.tight_layout()


from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
 
from dataset import DatasetGenerator
 
DIR = 'C:/Users/bijaw/Desktop/New folder (3)/trial'
 
INPUT_SHAPE = (177,98,1)
BATCH = 1
EPOCHS = 15
 
LABELS = 'bed cat happy'.split()
NUM_CLASSES = len(LABELS)


dsGen = DatasetGenerator(label_set=LABELS) 
# Load DataFrame with paths/labels 
df = dsGen.load_data(DIR)


dsGen.apply_train_test_split(test_size=0.0, random_state=2018)
dsGen.apply_train_val_split(val_size=0.01, random_state=2018)


from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
                          
def deep(features_shape, num_classes, act='relu'):

    # Input
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Flatten
    o = Flatten(name='flatten')(o)
    
    # Dense layer
    o = Dense(512, activation=act, name='dense1')(o)
    o = Dense(512, activation=act, name='dense2')(o)
    o = Dense(512, activation=act, name='dense3')(o)
    
    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)
    
    # Print network summary
    Model(inputs=x, outputs=o).summary()
    
    return Model(inputs=x, outputs=o)

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
 
def deep_cnn(features_shape, num_classes, act='relu'):
 
    x = Input(name='inputs', shape=features_shape, dtype='float32')
    o = x
    
    # Block 1
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block1_conv', input_shape=features_shape)(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block1_pool')(o)
    o = BatchNormalization(name='block1_norm')(o)
    
    # Block 2
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block2_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block2_pool')(o)
    o = BatchNormalization(name='block2_norm')(o)
 
    # Block 3
    o = Conv2D(32, (3, 3), activation=act, padding='same', strides=1, name='block3_conv')(o)
    o = MaxPooling2D((3, 3), strides=(2,2), padding='same', name='block3_pool')(o)
    o = BatchNormalization(name='block3_norm')(o)
 
    # Flatten
    o = Flatten(name='flatten')(o)
    
    # Dense layer
    o = Dense(64, activation=act, name='dense')(o)
    o = BatchNormalization(name='dense_norm')(o)
    o = Dropout(0.2, name='dropout')(o)
    
    # Predictions
    o = Dense(num_classes, activation='softmax', name='pred')(o)
 
    # Print network summary
    Model(inputs=x, outputs=o).summary()
    
    return Model(inputs=x, outputs=o)


model = deep_cnn(INPUT_SHAPE, NUM_CLASSES)
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])


callbacks = [EarlyStopping(monitor='val_acc', patience=4, verbose=1, mode='max')]
 
history = model.fit_generator(generator=dsGen.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(dsGen.df_train)/BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dsGen.generator(BATCH, mode='val'),
                              validation_steps=int(np.ceil(len(dsGen.df_val)/BATCH)))

#Reading audio from user
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 1
RATE = 16000 #sample rate
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "C:/Users/bijaw/Desktop/cat/new.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")

frames = []
#print(int(RATE / CHUNK * RECORD_SECONDS))

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")
#print(frames)
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
#end of reading audio

from scipy.io.wavfile import read
threshold_freq=5500 
eps=1e-10
wav = (read(WAVE_OUTPUT_FILENAME))[1]

from pydub import AudioSegment
from pydub.playback import play 
audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
play(audio)
# Sample rate
L = 16000
print(len(wav))
# If longer then randomly truncate
if len(wav) > L:
    i = np.random.randint(0, len(wav) - L)
    wav = wav[i:(i+L)]  
    print(len(wav))
# If shorter then randomly add silence
elif len(wav) < L:
    rem_len = L - len(wav)
    silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
    j = np.random.randint(0, rem_len)
    silence_part_left  = silence_part[0:j]
    silence_part_right = silence_part[j:rem_len]
    print(len(silence_part_left)," ",len(wav)," ",len(silence_part_right))
    wav = np.concatenate([silence_part_left, wav, silence_part_right])
# Create spectrogram using discrete FFT (change basis to frequencies)
freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
# Cut high frequencies
if threshold_freq is not None:
    spec = spec[freqs <= threshold_freq,:]
    freqs = freqs[freqs <= threshold_freq]
import pandas as pd
e = "cat/new.wav"
    
label, name = e.split('/')
label_id = dsGen.text_to_labels(label)
fle = "C:/Users/bijaw/Desktop/cat/new.wav"

sample = (label, label_id, name, fle)
n_df = pd.DataFrame(data = [sample],columns = ['label', 'label_id', 'user_id', 'wav_file'])
dsGen.df_test = dsGen.df_test.append(n_df)
df_new = dsGen.df_test
print("____DONE____")

'''
from pydub import AudioSegment
from pydub.playback import play

audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
play(audio)
'''

from keras.utils import to_categorical
import random
    
y_pred_proba = model.predict_generator(dsGen.generator(BATCH, mode='test'), 
                                     int(np.ceil(len(dsGen.df_test)/BATCH)), 
                                     verbose=1)

y_pred = np.argmax(y_pred_proba, axis=1)
 
y_true = dsGen.df_test['label_id'].values
 
print(dsGen.labels_to_text(y_pred[0]))


