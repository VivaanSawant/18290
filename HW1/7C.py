import numpy as np
import matplotlib.pyplot as plt


n = np.linspace(-20, +20, num=41, endpoint=True)
x = np.exp(n/5.0) * np.exp(-1j*np.pi*n/10.0)


xr = np.real(x)
xi = np.imag(x)
mag = np.abs(x)
ph = np.angle(x)

plt.figure(); 
plt.stem(n, xr); 
plt.title("Real"); 
plt.xlabel("n"); 
plt.figure(); 
plt.stem(n, xi); 
plt.title("Imaginary"); 
plt.xlabel("n");
plt.figure(); 
plt.stem(n, mag); 
plt.title("|x[n]|"); 
plt.xlabel("n");
plt.figure(); 
plt.stem(n, ph); 
plt.title("Angle x[n]"); 
plt.xlabel("n");
plt.show()