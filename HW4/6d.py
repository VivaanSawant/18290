import numpy as np
import matplotlib.pyplot as plt

W = 10

x = np.ones(W)

y = np.zeros(10 * W)
y[W:2*W] = 1
y[3*W:4*W] = 2
y[8*W:9*W] = 1

xr = x[::-1]

phi = np.convolve(xr, y, mode="full")

plt.figure()
plt.plot(phi)
plt.title("Cross-correlation φ_xy[k]")
plt.xlabel("index k")
plt.ylabel("value")
plt.show()

k_peak = np.argmax(phi)
tau_est = k_peak - (W - 1)

print("Peak index:", k_peak)
print("Estimated shift:", tau_est)
