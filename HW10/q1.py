import torch
import numpy as np

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