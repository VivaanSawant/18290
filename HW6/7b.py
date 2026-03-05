import numpy as np
import matplotlib.pyplot as plt

# some basic parameters
T0 = 1.0  # feel free to change this
T = 5.0   # feel free to change this

# start with a dense sampling of the time variable
tim = np.linspace(-1.5*T, 1.5*T, num=1000)

# the square wave
x_sq = np.abs(np.mod(tim + T0, T)) <= T0 * 2

# plot it
plt.subplots(1,1, figsize=(16,4))
plt.plot(tim, x_sq, label='Square wave')
plt.legend(loc='lower right')
plt.show()