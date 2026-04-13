import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

data = np.load(r'c:\Users\vivaa\OneDrive\Desktop\18290\HW9\hw9.npz')
xsharp = data['orca']

size = 15
kernel = (np.random.randn(size, size) > 2.0).astype(float)
kernel = kernel / np.sum(kernel)

y = scipy.signal.convolve2d(xsharp, kernel, mode='valid')
y = y + np.random.randn(*y.shape) / 200


def loss(x):
    conv = scipy.signal.convolve2d(x, kernel, mode='valid')
    return np.sum((conv - y) ** 2)


def gradient(x):
    conv = scipy.signal.convolve2d(x, kernel, mode='valid')
    error = conv - y
    grad = scipy.signal.convolve2d(error, np.flip(kernel), mode='full')
    return 2 * grad


x = np.zeros_like(xsharp)
lr = 0.01
iters = 300

losses = []

for i in range(iters):
    x = x - lr * gradient(x)
    losses.append(loss(x))

    if i % 10 == 0:
        print(f"Iter {i}: Loss = {losses[-1]:.4f}")


plt.imshow(x, cmap='gray')
plt.title('Recovered sharp image')
plt.axis('off')
plt.show()


plt.plot(losses)
plt.title('Loss function with iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()