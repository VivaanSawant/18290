import numpy as np
import matplotlib.pyplot as plt

T0 = 1.0
T = 5.0

t = np.linspace(-1.5*T, 1.5*T, num=1000)

x = np.abs(np.mod(t + T0, T)) <= T0 * 2

plt.subplots(1,1, figsize=(16,4))
plt.plot(t, x, label='Square wave')
plt.legend(loc='lower right')
plt.show()

w0 = 2*np.pi/T

def X(k):
    if k == 0:
        return 2*T0/T
    return np.sin(k*w0*T0)/(k*np.pi)

def xb(t,K):
    s = X(0)
    for k in range(1,K+1):
        s += 2*X(k)*np.cos(k*w0*t)
    return s