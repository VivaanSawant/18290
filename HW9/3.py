import numpy as np
import matplotlib.pyplot as plt

def symmetric_conv(x, h):
    N = len(x)
    L = len(h)
    y = np.zeros(N)
    
    center = (L - 1) // 2
    
    for n in range(N):
        for k in range(L):
            idx = n - (k - center)
            
            if idx < 0:
                idx = -idx - 1
            elif idx >= N:
                idx = 2*N - idx - 1
            
            y[n] += h[k] * x[idx]
    
    return y


# Signals
n = np.linspace(0,10,11)
x = np.exp(-n/5) + np.exp(-(10-n)/5)
h = np.array([1.0, -1.0])

# Outputs
y_sym = symmetric_conv(x, h)
y_zero = np.convolve(x, h, mode='same')


# -------- Plot 1: Symmetric Boundary --------
plt.figure()
plt.plot(y_sym)
plt.title("Convolution with Symmetric Boundary Conditions")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid()
plt.show()


# -------- Plot 2: Zero Boundary --------
plt.figure()
plt.plot(y_zero)
plt.title("Convolution with Zero Boundary Conditions")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid()
plt.show()