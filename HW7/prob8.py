import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs = 96000
duration = 5

print("Recording message...")
message = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()

message = message.flatten()

print("Playback original message")
sd.play(message, fs)
sd.wait()

# Compute Fourier Transform
N = len(message)
spectrum = np.fft.fft(message)
freqs = np.fft.fftfreq(N, 1/fs)

# Plot magnitude spectrum
plt.figure()
plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of Recorded Message")
plt.show()


cutoff = 1000

b, a = butter(6, cutoff/(fs/2), btype='low')

filtered_message = lfilter(b, a, message)

# FFT
N = len(filtered_message)
spectrum = np.fft.fft(filtered_message)
freqs = np.fft.fftfreq(N, 1/fs)

plt.figure()
plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of Lowpass Filtered Message")
plt.show()

fc = 15000

t = np.arange(len(filtered_message)) / fs
carrier = np.cos(2 * np.pi * fc * t)

tx_signal = filtered_message * carrier

# FFT
N = len(tx_signal)
spectrum = np.fft.fft(tx_signal)
freqs = np.fft.fftfreq(N, 1/fs)

plt.figure()
plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of Transmitted Signal")
plt.show()

print("Transmitting and receiving signal...")

rx_signal = sd.playrec(tx_signal, samplerate=fs, channels=1)
sd.wait()

rx_signal = rx_signal.flatten()

# FFT
N = len(rx_signal)
spectrum = np.fft.fft(rx_signal)
freqs = np.fft.fftfreq(N, 1/fs)

plt.figure()
plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of Received Signal")
plt.show()

demodulated = rx_signal * carrier

recovered = lfilter(b, a, demodulated)

# FFT
N = len(recovered)
spectrum = np.fft.fft(recovered)
freqs = np.fft.fftfreq(N, 1/fs)

plt.figure()
plt.plot(freqs[:N//2], np.abs(spectrum[:N//2]))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Spectrum of Demodulated Signal")
plt.show()