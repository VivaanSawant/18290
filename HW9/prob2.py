import numpy as np

x = np.array([1,2,3,4,5,6,7,8], dtype=float)
h = np.array([1,-1,2,-1], dtype=float)

# Linear convolution
y_linear = np.convolve(x, h)

# Circular convolution (length Nx)
Nx = len(x)
y_circ = np.fft.ifft(np.fft.fft(x, Nx) * np.fft.fft(h, Nx)).real

print("Linear:", y_linear)
print("Circular:", y_circ)

def dft_convolution(x, h, M):
    # Step 1
    X = np.fft.fft(x, n=M)  # startx
    H = np.fft.fft(h, n=M)
    
    # Step 2
    Y = X * H
    
    # Step 3
    y = np.fft.ifft(Y, n=M).real
    return y

M = len(x) + len(h) - 1 + 4
y = dft_convolution(x, h, M)

print("y:", y)