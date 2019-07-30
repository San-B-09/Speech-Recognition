import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 1
RATE = 16000 #sample rate
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "C:\\Users\\bijaw\\Desktop\\audio.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")

frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

from pydub import AudioSegment
from pydub.playback import play
import wave

print("* playing")
audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
play(audio)
    
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import stft  

threshold_freq=5500 
eps=1e-10
wav = (read(WAVE_OUTPUT_FILENAME))[1]

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



from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=100,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
)
#audio = AudioSegment.from_wav(wav)
for i, chunk in enumerate(audio_chunks):

    out_file = "C:/Users/bijaw/Desktop/chunk{0}.wav".format(i)
    print ("exporting", out_file)
    chunk.export(out_file, format="wav")

    