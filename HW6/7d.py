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

def xb(v,K):
    s = X(0)*np.ones_like(v)
    for k in range(1,K+1):
        s += 2*X(k)*np.cos(k*w0*v)
    return s

x5 = xb(t,5)
x10 = xb(t,10)
x100 = xb(t,100)

plt.subplots(1,1, figsize=(16,4))
plt.plot(t, x, label='Square wave')
plt.plot(t, x5, label='K=5')
plt.plot(t, x10, label='K=10')
plt.plot(t, x100, label='K=100')
plt.legend(loc='lower right')
plt.show()