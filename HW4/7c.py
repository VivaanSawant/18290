import numpy as np
import matplotlib.pyplot as plt

data = np.load("hw4.npz")
spike01 = data["spike01"]

Ns = [10, 25, 50, 100, 200]

plt.figure()

for N in Ns:
    W = np.ones(N) / N

    rate_est = np.convolve(spike01, W, mode="same")

    plt.plot(rate_est, label=f"N = {N}")

plt.legend()
plt.title("Effect of Impulse Response Length on Rate Estimate")
plt.xlabel("Time")
plt.ylabel("Estimated Firing Rate")
plt.show()
