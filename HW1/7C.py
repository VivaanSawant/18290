import numpy as np
import matplotlib.pyplot as plt
n = np.linspace(-20, +20, num=41, endpoint=True)
x = np.exp(n/5.0) * np.exp(-1j*np.pi*n/10.0)
