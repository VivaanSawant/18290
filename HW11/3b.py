import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom

def function_H(x, x_siz, theta):
    x = x.reshape(x_siz)
    return radon(x, theta=theta)

def function_H_transpose(y, y_siz, theta):
    y = y.reshape(y_siz)
    return iradon(y, theta=theta, filter_name=None)

def cgs(A, b, x0, max_iter=20, tol=1e-6):
    x = x0.copy()
    r = b - A(x)
    p = r.copy()
    rsold = np.dot(r.flatten(), r.flatten())
    errs = []
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / np.dot(p.flatten(), Ap.flatten())
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r.flatten(), r.flatten())
        errs.append(np.sqrt(rsnew))
        print("Part B Iteration", i, "Residual", np.sqrt(rsnew), flush=True)
        if np.sqrt(rsnew) < tol:
            print("Part B Converged at iteration", i, flush=True)
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x, errs

x_true = shepp_logan_phantom()
x_siz = x_true.shape

theta = np.linspace(0., 180., 30, endpoint=False)

y0 = function_H(x_true, x_siz, theta)
y0 = y0 + np.random.normal(0, 0.05, size=y0.shape)
y_siz = y0.shape

def A(x):
    return function_H_transpose(function_H(x, x_siz, theta), y_siz, theta)

b = function_H_transpose(y0, y_siz, theta)

x_init = np.zeros(x_siz)

x_star, errs = cgs(A, b, x_init)

print("Part B Output: Reconstructed image x_star")
print(x_star)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
ax1.imshow(x_true, cmap='gray')
ax1.set_title("Part B Ground Truth Image x_true")
ax2.imshow(x_star, cmap='gray')
ax2.set_title("Part B Reconstructed Image x_star")
plt.show()