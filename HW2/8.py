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


h1 = np.array([1, -1])

imgoutput_b = np.zeros_like(img)

for rr in range(img.shape[0]):
    imgoutput_b[rr, :] = convolve(img[rr, :], h1, mode='same')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(imgoutput_b, cmap='gray')
axes[1].set_title('Convolved with h1 = [1, -1]')
axes[1].axis('off')

plt.savefig('Convolution_b.png')
plt.show()


h2 = np.ones(10)

imgoutput_c = np.zeros_like(img)

for rr in range(img.shape[0]):
    imgoutput_c[rr, :] = convolve(img[rr, :], h2, mode='same')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(imgoutput_c, cmap='gray')
axes[1].set_title('Convolved with h2')
axes[1].axis('off')

plt.savefig('Convolution_c.png')
plt.show()



D = 20 

#h[n] = delta[n] + delta[n-D]
h = np.zeros(D + 1)
h[0] = 1
h[D] = 1

output = np.zeros_like(img)

for rr in range(img.shape[0]):
    output[rr, :] = convolve(img[rr, :], h, mode='same')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(output, cmap='gray')
axes[1].set_title(f'Shifted copy added (D = {D})')
axes[1].axis('off')

plt.savefig('Convolution_d.png')
plt.show()
