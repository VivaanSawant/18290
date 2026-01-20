import sounddevice as sd
import numpy as np
fs = 44100 # Sample rate (samples per second)
duration = 10 # Duration in seconds
channels = 1 # We are going for mono recording
num = int(duration * fs)
# Recording
print(f'Recording for {duration} seconds. say something')
rec = sd.rec(num, samplerate=fs, channels=channels, dtype='float64')
sd.wait()
print('Recording complete.')
# Playback
print('Playing recorded audio...')
sd.play(rec, fs)
sd.wait()
print('Playback complete.')
