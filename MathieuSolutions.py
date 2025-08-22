from scipy.special import mathieu_cem, mathieu_sem
import numpy as np
import matplotlib.pyplot as plt

# Parameters from your problem
a = 0.01
q = 0.2
Omega = 10**7
omega_rf = 10**6

a_param = a + 4*Omega**2 / omega_rf**2   # maps to "a" in mathieu functions

# Evaluate even/odd Mathieu functions at some tau array
tau_vals = np.linspace(0, 20, 500)
y_even = mathieu_cem(0, q, tau_vals)[0]  # even solution, order=0
y_odd  = mathieu_sem(1, q, tau_vals)[0] # odd solution, order=1

fig, ax = plt.subplots(1, 2)

ax[0].plot(tau_vals, y_even)
ax[0].set_title("Even solution")

ax[1].plot(tau_vals, y_odd)
ax[1].set_title("Odd solution")

plt.tight_layout()
plt.show()
