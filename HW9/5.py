import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
data = np.load(r'c:\Users\vivaa\OneDrive\Desktop\18290\HW9\hw9.npz')
orca = data['orca']

# Normalize target (important)
orca = orca / np.linalg.norm(orca)


# -------------------------------
# Gerchberg-Saxton 2D Algorithm
# -------------------------------
def gerchberg_saxton_2d(Itar, max_iter=100):
    
    Ik = Itar.copy()
    err = []
    
    for k in range(max_iter):
        
        # Step 1: get phase of current estimate
        Iang = np.angle(Ik)
        
        # Step 2: enforce target magnitude (frequency domain)
        Ib = Itar * np.exp(1j * Iang)
        
        # Step 3: inverse FFT to go to lens
        ub = np.fft.ifft2(Ib)
        
        # Step 4: extract phase
        phi = np.angle(ub)
        
        # Step 5: enforce phase-only constraint
        wb = np.exp(1j * phi)
        
        # Step 6: FFT back to retina
        Ik = np.fft.fft2(wb)
        
        # Step 7: normalize (important for stability)
        Ik = Ik / np.linalg.norm(Ik)
        
        # Step 8: compute error
        err.append(np.linalg.norm(Itar - np.abs(Ik)))
    
    return np.abs(Ik), phi, err


# -------------------------------
# Run algorithm
# -------------------------------
output2d, phi2d, err2d = gerchberg_saxton_2d(orca, 100)


# -------------------------------
# Plot results
# -------------------------------

# Target Image
plt.figure()
plt.imshow(orca, cmap='gray')
plt.title("Target Image")
plt.colorbar()
plt.axis('off')
plt.show()

# Output Image
plt.figure()
plt.imshow(output2d, cmap='gray')
plt.title("Output Image (After GS Algorithm)")
plt.colorbar()
plt.axis('off')
plt.show()

# Error Plot
plt.figure()
plt.plot(err2d)
plt.title("Error vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.grid()
plt.show()