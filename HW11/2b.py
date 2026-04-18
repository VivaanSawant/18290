import numpy as np
import matplotlib.pyplot as plt

def sample_identity_rows(N, M):
    indices = np.random.choice(N, size=M, replace=False)
    return np.eye(N)[indices]

def cgs(A, b, x0, max_iter=100, tol=1e-6):
    x = x0.copy()
    r = b - A(x)
    p = r.copy()
    rsold = np.dot(r, r)
    errs = []

    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        errs.append(np.sqrt(rsnew))
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return x, errs

N = 128
M = 64
lam = 0.5

H = sample_identity_rows(N, M)

x0 = np.cumsum(np.random.randn(N)/5)

y = H @ x0 + 0.1 * np.random.randn(M)

def A(x):
    return H.T @ (H @ x) + lam * x

b = H.T @ y

x_init = np.zeros(N)

x_star, errs = cgs(A, b, x_init)

print("Part B: Reconstructed signal x_star with L2 regularization")
print(x_star)

plt.figure()
plt.plot(x0, label="Ground Truth x0")
plt.plot(x_star, label="Reconstructed x_star")
plt.legend()
plt.title("Part B Reconstruction")
plt.show()