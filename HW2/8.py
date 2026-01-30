# we will need some packages
# if they are not installed, then use conda or pip3 to install them
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage import data
img = data.camera().astype(float) / 255.0
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set title('Original Image')
for rr in range(img.shape[0]):
axes[1].plot(img[rr, :])
axes[1].set title('Rows plotted as 1D signals')
plt.savefig('Row 1D signals.png')
plt.show()
