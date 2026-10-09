#%% 
# %% Cell 1: Imports and computational grid

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigsh
# Dimensions of the rectangular billiard
Lx = 2.0
Ly = 1.0

# Number of interior grid points
Nx = 60
Ny = 30

# Grid spacing
dx = Lx / (Nx + 1)
dy = Ly / (Ny + 1)

# Interior coordinates
x = np.linspace(dx, Lx - dx, Nx)
y = np.linspace(dy, Ly - dy, Ny)

# Create a two-dimensional grid
X, Y = np.meshgrid(x, y)

# Display the computational grid
plt.figure(figsize=(8, 4))
plt.scatter(X, Y, s=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rectangular Quantum Billiard Grid")
plt.axis("equal")
plt.show()

print(f"Grid points: {Nx * Ny}")
print(f"dx = {dx:.5f}")
print(f"dy = {dy:.5f}")



# %%

# %% Cell 2: Build the Hamiltonian matrix

# Total number of interior grid points
N = Nx * Ny

# Create an empty sparse matrix
H = lil_matrix((N, N))

# Loop over every interior grid point
for j in range(Ny):
    for i in range(Nx):

        # Convert 2D grid position to 1D matrix index
        k = j * Nx + i

        # Diagonal term
        H[k, k] = 2 / dx**2 + 2 / dy**2

        # Left neighbor
        if i > 0:
            H[k, k - 1] = -1 / dx**2

        # Right neighbor
        if i < Nx - 1:
            H[k, k + 1] = -1 / dx**2

        # Bottom neighbor
        if j > 0:
            H[k, k - Nx] = -1 / dy**2

        # Top neighbor
        if j < Ny - 1:
            H[k, k + Nx] = -1 / dy**2

# Convert to CSR format for efficient calculations
H = H.tocsr()

print("Hamiltonian constructed successfully!")
print("Matrix shape:", H.shape)
print("Nonzero elements:", H.nnz)


# %% Cell 3: Solve the Schrodinger equation

# Number of eigenstates to calculate
num_states = 10

# Solve H psi = E psi
energies, wavefunctions = eigsh(
    H,
    k=num_states,
    which="SM"
)

# Sort the eigenvalues from lowest to highest
order = np.argsort(energies)
energies = energies[order]
wavefunctions = wavefunctions[:, order]

# Print the calculated energy levels
print("Numerical Energy Eigenvalues:")
for n, E in enumerate(energies):
    print(f"State {n+1}: E = {E:.5f}")

# Exact ground-state energy for a rectangle
E_exact = np.pi**2 * (1/Lx**2 + 1/Ly**2)

print(f"\nExact ground-state energy: {E_exact:.5f}")
print(f"Numerical ground-state energy: {energies[0]:.5f}")
print(f"Error: {abs(energies[0]-E_exact):.5f}")

# %%

# %% Cell 4: Quarter-stadium geometry and masked Hamiltonian

# Quarter-stadium: rectangular section [0, a] x [0, R] plus a semicircular cap
a = 1.0   # half-length of straight section
R = 1.0   # radius of semicircular caps

# Grid covering the bounding box [0, a+R] x [0, R]
Lx_s = a + R
Ly_s = R
Nx_s = 120
Ny_s = 60

dx_s = Lx_s / (Nx_s + 1)
dy_s = Ly_s / (Ny_s + 1)

x_s = np.linspace(dx_s, Lx_s - dx_s, Nx_s)
y_s = np.linspace(dy_s, Ly_s - dy_s, Ny_s)
X_s, Y_s = np.meshgrid(x_s, y_s)

# A point is inside the quarter-stadium if it is in the rectangular section
# OR inside the right semicircular cap
mask = (X_s <= a) | ((X_s - a)**2 + Y_s**2 <= R**2)

plt.figure(figsize=(8, 4))
plt.imshow(mask, origin="lower", extent=[0, Lx_s, 0, Ly_s], cmap="Blues")
plt.title("Quarter-Stadium Mask")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
print(f"Interior points: {mask.sum()} / {Nx_s * Ny_s}")

# Map each (j, i) grid point inside the mask to a compact 1D index
flat_mask = mask.flatten()
N_s = flat_mask.sum()
idx = -np.ones(Nx_s * Ny_s, dtype=int)
idx[flat_mask] = np.arange(N_s)

# Build the Hamiltonian — same stencil as before, but skip exterior points
H_s = lil_matrix((N_s, N_s))

for j in range(Ny_s):
    for i in range(Nx_s):
        kf = j * Nx_s + i
        if not flat_mask[kf]:
            continue
        k = idx[kf]

        H_s[k, k] = 2 / dx_s**2 + 2 / dy_s**2

        if i > 0 and flat_mask[j * Nx_s + (i - 1)]:
            H_s[k, idx[j * Nx_s + (i - 1)]] = -1 / dx_s**2

        if i < Nx_s - 1 and flat_mask[j * Nx_s + (i + 1)]:
            H_s[k, idx[j * Nx_s + (i + 1)]] = -1 / dx_s**2

        if j > 0 and flat_mask[(j - 1) * Nx_s + i]:
            H_s[k, idx[(j - 1) * Nx_s + i]] = -1 / dy_s**2

        if j < Ny_s - 1 and flat_mask[(j + 1) * Nx_s + i]:
            H_s[k, idx[(j + 1) * Nx_s + i]] = -1 / dy_s**2

H_s = H_s.tocsr()
print("Stadium Hamiltonian shape:", H_s.shape)
print("Nonzero elements:", H_s.nnz)


# %% Cell 5: Solve for stadium eigenstates

num_states_s = 200

energies_s, wavefunctions_s = eigsh(H_s, k=num_states_s, which="SM")
order_s = np.argsort(energies_s)
energies_s = energies_s[order_s]
wavefunctions_s = wavefunctions_s[:, order_s]

print(f"Solved {num_states_s} stadium eigenstates")
print(f"Energy range: {energies_s[0]:.3f} to {energies_s[-1]:.3f}")


# %% Cell 6: Visualize the first 20 stadium eigenstates

fig, axes = plt.subplots(4, 5, figsize=(15, 10))

for n, ax in enumerate(axes.flat):
    psi = np.zeros(Ny_s * Nx_s)
    psi[flat_mask] = wavefunctions_s[:, n]
    psi_sq = psi.reshape(Ny_s, Nx_s) ** 2
    psi_sq[~mask] = np.nan

    ax.imshow(psi_sq, origin="lower",
              extent=[0, Lx_s, 0, Ly_s],
              cmap="inferno", interpolation="bilinear")
    ax.set_title(f"n={n+1}, E={energies_s[n]:.2f}", fontsize=8)
    ax.axis("off")

plt.suptitle("Quarter-Stadium: First 20 Eigenstates |ψ|²")
plt.tight_layout()
plt.show()

# %%# %% Cell 9: Full stadium (both symmetry classes mixed — level statistics will look wrong)

a_f = 1.0
R_f = 1.0

# Bounding box: [0, 2(a+R)] x [0, 2R]
Lx_f = 2 * (a_f + R_f)
Ly_f = 2 * R_f
Nx_f = 240
Ny_f = 120

dx_f = Lx_f / (Nx_f + 1)
dy_f = Ly_f / (Ny_f + 1)

x_f = np.linspace(dx_f, Lx_f - dx_f, Nx_f)
y_f = np.linspace(dy_f, Ly_f - dy_f, Ny_f)
X_f, Y_f = np.meshgrid(x_f, y_f)

# Centered coordinates
xc = X_f - (a_f + R_f)
yc = Y_f - R_f

# Distance from each point to the horizontal segment [-a, a] on the x-axis
# (this is the cleanest way to define a stadium)
dist_to_segment = np.sqrt(np.maximum(np.abs(xc) - a_f, 0)**2 + yc**2)
mask_f = dist_to_segment <= R_f

plt.figure(figsize=(10, 4))
plt.imshow(mask_f, origin="lower", extent=[0, Lx_f, 0, Ly_f], cmap="Blues")
plt.title("Full Stadium Mask")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.show()
print(f"Interior points: {mask_f.sum()} / {Nx_f * Ny_f}")

flat_mask_f = mask_f.flatten()
N_f = flat_mask_f.sum()
idx_f = -np.ones(Nx_f * Ny_f, dtype=int)
idx_f[flat_mask_f] = np.arange(N_f)

H_f = lil_matrix((N_f, N_f))

for j in range(Ny_f):
    for i in range(Nx_f):
        kf = j * Nx_f + i
        if not flat_mask_f[kf]:
            continue
        k = idx_f[kf]

        H_f[k, k] = 2 / dx_f**2 + 2 / dy_f**2

        if i > 0 and flat_mask_f[j * Nx_f + (i - 1)]:
            H_f[k, idx_f[j * Nx_f + (i - 1)]] = -1 / dx_f**2

        if i < Nx_f - 1 and flat_mask_f[j * Nx_f + (i + 1)]:
            H_f[k, idx_f[j * Nx_f + (i + 1)]] = -1 / dx_f**2

        if j > 0 and flat_mask_f[(j - 1) * Nx_f + i]:
            H_f[k, idx_f[(j - 1) * Nx_f + i]] = -1 / dy_f**2

        if j < Ny_f - 1 and flat_mask_f[(j + 1) * Nx_f + i]:
            H_f[k, idx_f[(j + 1) * Nx_f + i]] = -1 / dy_f**2

H_f = H_f.tocsr()
print("Full stadium Hamiltonian shape:", H_f.shape)


# %% Cell 10: Solve and visualize full stadium

num_states_f = 200

energies_f, wavefunctions_f = eigsh(H_f, k=num_states_f, which="SM")
order_f = np.argsort(energies_f)
energies_f = energies_f[order_f]
wavefunctions_f = wavefunctions_f[:, order_f]

ipr_f = np.array([
    np.sum(wavefunctions_f[:, n]**4) * dx_f * dy_f
    for n in range(num_states_f)
])
scar_order_f = np.argsort(ipr_f)[::-1]

fig, axes = plt.subplots(4, 5, figsize=(15, 8))

for rank, ax in enumerate(axes.flat):
    n = scar_order_f[rank]
    psi = np.zeros(Ny_f * Nx_f)
    psi[flat_mask_f] = wavefunctions_f[:, n]
    psi_sq = psi.reshape(Ny_f, Nx_f) ** 2
    psi_sq[~mask_f] = np.nan

    ax.imshow(psi_sq, origin="lower",
              extent=[0, Lx_f, 0, Ly_f],
              cmap="inferno", interpolation="bilinear")
    ax.set_title(f"n={n+1}, IPR={ipr_f[n]:.4f}", fontsize=8)
    ax.axis("off")

plt.suptitle("Full Stadium: 20 Most Localized States (note: symmetry classes mixed)")
plt.tight_layout()
plt.show()

# %% 
plt.suptitle("Quarter-Stadium Scars with Classical Orbit Overlay")
plt.tight_layout()
plt.show()


# %% Cell 13: Rectangle with irrational aspect ratio and 200 states

Lx_r = np.sqrt(2)
Ly_r = 1.0
Nx_r = 80
Ny_r = 57

dx_r = Lx_r / (Nx_r + 1)
dy_r = Ly_r / (Ny_r + 1)

x_r = np.linspace(dx_r, Lx_r - dx_r, Nx_r)
y_r = np.linspace(dy_r, Ly_r - dy_r, Ny_r)

N_r = Nx_r * Ny_r
H_r = lil_matrix((N_r, N_r))

for j in range(Ny_r):
    for i in range(Nx_r):
        k = j * Nx_r + i
        H_r[k, k] = 2 / dx_r**2 + 2 / dy_r**2
        if i > 0:
            H_r[k, k - 1] = -1 / dx_r**2
        if i < Nx_r - 1:
            H_r[k, k + 1] = -1 / dx_r**2
        if j > 0:
            H_r[k, k - Nx_r] = -1 / dy_r**2
        if j < Ny_r - 1:
            H_r[k, k + Nx_r] = -1 / dy_r**2

H_r = H_r.tocsr()

num_states_r = 200
energies_r, wavefunctions_r = eigsh(H_r, k=num_states_r, which="SM")
order_r = np.argsort(energies_r)
energies_r = energies_r[order_r]
wavefunctions_r = wavefunctions_r[:, order_r]

dA_r = dx_r * dy_r
ipr_r = np.array([np.sum(wavefunctions_r[:, n]**4) * dA_r for n in range(num_states_r)])

print(f"Rectangle (Lx=√2): {num_states_r} states solved")
print(f"Mean IPR: {ipr_r.mean():.6f},  Std: {ipr_r.std():.6f}")


# %% Cell 14: Statistical comparison — IPR distribution and level spacing

from scipy.stats import gaussian_kde

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: IPR histogram ---
ax = axes[0]
bins = np.linspace(0, max(ipr_r.max(), ipr.max()) * 1.05, 40)
ax.hist(ipr_r, bins=bins, alpha=0.6, color="steelblue", label="Rectangle (integrable)")
ax.hist(ipr,   bins=bins, alpha=0.6, color="tomato",    label="Quarter-Stadium (chaotic)")
ax.set_xlabel("IPR")
ax.set_ylabel("Count")
ax.set_title("IPR Distribution")
ax.legend()

# --- Right panel: nearest-neighbor level spacing ---
ax = axes[1]

def level_spacings(energies):
    """Unfold eigenvalues so mean spacing = 1, return nearest-neighbor gaps."""
    # Smooth the staircase with a polynomial fit to get the mean density
    n_fit = np.arange(len(energies))
    coeffs = np.polyfit(energies, n_fit, deg=5)
    unfolded = np.polyval(coeffs, energies)
    spacings = np.diff(unfolded)
    return spacings / spacings.mean()

s_rect  = level_spacings(energies_r)
s_stad  = level_spacings(energies_s)

s_range = np.linspace(0, 4, 300)
poisson     = np.exp(-s_range)                            # integrable prediction
wigner      = (np.pi / 2) * s_range * np.exp(-np.pi / 4 * s_range**2)  # chaotic prediction

ax.hist(s_rect, bins=30, density=True, alpha=0.6, color="steelblue", label="Rectangle")
ax.hist(s_stad, bins=30, density=True, alpha=0.6, color="tomato",    label="Quarter-Stadium")
ax.plot(s_range, poisson, "b--", lw=2, label="Poisson (integrable)")
ax.plot(s_range, wigner,  "r-",  lw=2, label="Wigner-Dyson (chaotic)")
ax.set_xlabel("s  (normalized spacing)")
ax.set_ylabel("P(s)")
ax.set_title("Nearest-Neighbor Level Spacing")
ax.legend()

plt.suptitle("Spectral Statistics: Integrable vs. Chaotic Billiard")
plt.tight_layout()
plt.show()