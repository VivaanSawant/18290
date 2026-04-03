import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# Load audio
x, fs = sf.read(r"C:\Users\vivaa\OneDrive\Desktop\18290\HW8\rumblestrip.wav")

# Convert to mono if stereo
if x.ndim > 1:
    x = x[:, 0]

N = len(x)
n = np.arange(N)

# -----------------------------
# Compute DTFT
# -----------------------------
K = 1000
w = np.linspace(-np.pi/8, np.pi/8, K)

X = np.array([
    np.sum(x * np.exp(-1j * wi * n))
    for wi in w
])

# -----------------------------
# Plot DTFT
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(w, np.abs(X))
plt.title("DTFT Magnitude |X(e^{jΩ})|")
plt.xlabel("Omega (rad/sample)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Find fundamental frequency
# -----------------------------
k_max = np.argmax(np.abs(X))
w0 = abs(w[k_max])   # take magnitude (positive frequency)

# -----------------------------
# Compute period N0
# -----------------------------
N0 = 2 * np.pi / w0

print("Omega_0 (rad/sample):", w0)
print("N0 (samples):", N0)

# -----------------------------
# Convert to Hz
# -----------------------------
f0 = (w0 * fs) / (2 * np.pi)
print("f0 (Hz):", f0)

# -----------------------------
# Rumble strip spacing (meters)
# -----------------------------
d = 0.317   # from your measurement

# -----------------------------
# Compute speed
# -----------------------------
v = f0 * d
mph = v * 2.237

print("Speed (m/s):", v)
print("Speed (mph):", mph)

# -----------------------------
# Speeding check
# -----------------------------
if mph > 50:
    print("The car WAS speeding.")
else:
    print("The car was NOT speeding.")