import torch
import numpy as np
import matplotlib.pyplot as plt

def optimize_phase(Itar, num_iters=300, lr=0.05):

    Itar = Itar / torch.norm(Itar)

    phi = torch.rand_like(Itar) * 2 * np.pi
    phi.requires_grad_(True)

    opt = torch.optim.Adam([phi], lr=lr)

    losses = []

    for _ in range(num_iters):
        signal = torch.exp(1j * phi)
        spectrum = torch.fft.fft(signal)
        mag = torch.abs(spectrum)
        mag = mag / torch.norm(mag)

        loss = torch.norm(Itar - mag)**2

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    return phi.detach(), mag.detach(), losses


data = np.load(r'c:\Users\vivaa\OneDrive\Desktop\18290\HW9\hw9.npz')
Itar_np = data['orca']

Itar = torch.tensor(Itar_np, dtype=torch.float32)
Itar = Itar / torch.norm(Itar)

phi_opt, output_mag, losses = optimize_phase(Itar)


plt.figure()
plt.plot(output_mag.numpy())
plt.title('Image it produces')
plt.xlabel('Index')
plt.ylabel('Magnitude')
plt.show()


plt.figure()
plt.plot(losses)
plt.title('Loss function with iteration')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


print("Optimized Phase Pattern:")
print(phi_opt)