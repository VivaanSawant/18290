# we will need some packages
# if they are not installed, then use conda or pip3 to install them
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from skimage import data

# load image and normalize
img = data.camera().astype(float) / 255.0

# create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# show original image
axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# plot each row as a 1D signal
for rr in range(img.shape[0]):
    axes[1].plot(img[rr, :])

axes[1].set_title('Rows plotted as 1D signals')
axes[1].set_xlabel('Column index')
axes[1].set_ylabel('Intensity')

plt.savefig('Row_1D_signals.png')
plt.show()
