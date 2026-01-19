# Imports
import numpy as np
import matplotlib.pyplot as plt

# Simplified dual beam setup for LG01 beams

# range
y = np.linspace(-3, 3, 400)
z = np.linspace(-3, 3, 400)
Y, Z = np.meshgrid(y, z)
r = np.sqrt(Y**2 + Z**2)
phi = np.arctan2(Y, Z)

# Parameters
w = 1.0      # beam waist
k = 2*np.pi  # wave number (wavelength=1)
x = 0        # x = 0 plane

# LG10 mode, two counter propagating beams
E_plus  = (r/w) * np.exp(-r**2 / w**2) * np.exp(1j*phi) * np.exp(1j*k*x)
E_minus = (r/w) * np.exp(-r**2 / w**2) * np.exp(1j*phi) * np.exp(-1j*k*x)

# Total intensity
I = np.abs(E_plus + E_minus)**2

# add a point to intensity minimum on the optical axis
center_y, center_z = 0, 0

# Plot 
plt.figure(figsize=(6,6))
plt.scatter(center_y, center_z, color = 'red', s=10)
plt.imshow(I, extent=[y.min(), y.max(), z.min(), z.max()],
           origin='lower', cmap='viridis')
plt.colorbar(label='Intensity')
plt.xlabel('y')
plt.ylabel('z')
plt.title('Cross-section of LG bottle beam at x=0 on the optical axis')
plt.show()
