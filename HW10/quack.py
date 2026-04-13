import numpy as np

h = np.array([1,1,1,0,0,0,0,0])

N = len(h)
H = np.zeros((N, N))

for i in range(N):
    H[i] = np.roll(h, i)

print(H)

import numpy as np
import scipy.linalg

h = np.array([1,1,1,0,0,0,0,0])

N = len(h)
H = np.zeros((N, N))

for i in range(N):
    H[i] = np.roll(h, i)

F = scipy.linalg.dft(N)
F_inv = np.linalg.inv(F)

C = F @ H @ F_inv

print(np.round(np.abs(C), 5))