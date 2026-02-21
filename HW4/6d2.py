import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


fs = 44100  
f0 = 100 
f1 = 1000  
amp = 1.0  
T = 1.0    
c = (f1 - f0) / (2.0 * T)  

nSamples = int(T * fs)   
tim = np.linspace(0, T, nSamples)

chirp = amp * np.cos(2 * np.pi * tim * (f0 + c * tim))

sd.play(chirp, fs)
sd.wait()

plt.subplot(1, 1, 1)
plt.plot(chirp[:2000])
plt.show()


fs = 1000
tim = np.linspace(0, 1, num=fs)

f0 = 100
f1 = 500
T = 0.7
c = (f1 - f0) / (2.0 * T)
x = np.cos(2 * np.pi * tim * (f0 + c * tim))
L = len(x)

y = np.zeros((2 * L,))
y[100:100 + L] = x            
y[500:500 + L] += x + 2.0   
y[900:900 + L] += 4 * x      

y_noisy = y + np.random.randn(*y.shape) / 2.0

plt.plot(y, label='y')
plt.plot(y_noisy, label='y-noisy')
plt.legend()
plt.show()
