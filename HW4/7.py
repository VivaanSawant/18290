import numpy as np
import matplotlib.pyplot as plt

data = np.load("hw4.npz")
spike01 = data["spike02"]

N = 50   
W = np.ones(N) / N

rate_est = np.convolve(spike01, W, mode="same")

dr = np.diff(rate_est)
dr = np.concatenate(([0], dr))

threshold = 0.02
change_idx = np.where(np.abs(dr) > threshold)[0]

plt.figure()
plt.stem(rate_est)
plt.title("Estimated Firing Rate (spike02)")
plt.xlabel("Time")
plt.ylabel("Rate")
plt.show()

plt.figure()
plt.stem(dr)
plt.axhline(threshold, color="r", linestyle="--")
plt.axhline(-threshold, color="r", linestyle="--")
plt.title("Rate Change Detector")
plt.xlabel("Time")
plt.ylabel("Δ Rate")
plt.show()

if len(change_idx) > 0:
    print("Stimulus change detected at index:", change_idx[0])
else:
    print("No stimulus change detected.")
