# %%
import numpy as np
import matplotlib.pyplot as plt


class IsingModel:
    def __init__(self, L, dim=2, J=1.0, T=2.0, h=0.0,
                 start="random", seed=None):
        if dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        if L < 3:
            raise ValueError("Use L >= 3")
        if T <= 0:
            raise ValueError("Temperature must be positive")

        self.L = L
        self.dim = dim
        self.J = J
        self.T = T                 # units where k_B = 1
        self.h = h                 # external field
        self.shape = (L,) * dim
        self.N = L ** dim
        self.rng = np.random.default_rng(seed)

        # Each spin is either -1 or +1.
        if start == "up":
            self.spins = np.ones(self.shape, dtype=int)
        elif start == "random":
            self.spins = self.rng.choice([-1, 1], size=self.shape)
        else:
            raise ValueError("start must be 'up' or 'random'")

        self.M = self.spins.sum()
        self.E = self.total_energy()

    def neighbor_sum(self, site):
        """Sum the four (2D) or six (3D) neighboring spins."""
        total = 0

        for axis in range(self.dim):
            for direction in (-1, 1):
                neighbor = list(site)
                neighbor[axis] = (neighbor[axis] + direction) % self.L
                total += self.spins[tuple(neighbor)]

        return total

    def total_energy(self):
        """Count each nearest-neighbor bond once, plus the field term."""
        energy = 0.0

        for axis in range(self.dim):
            energy -= self.J * np.sum(
                self.spins * np.roll(self.spins, -1, axis=axis)
            )

        energy -= self.h * np.sum(self.spins)

        return energy

    def metropolis_sweep(self):
        """Attempt N random single-spin flips with the Metropolis rule."""
        for _ in range(self.N):
            site = tuple(self.rng.integers(self.L, size=self.dim))
            s = self.spins[site]

            # Energy change from flipping this spin.
            dE = 2 * s * (self.J * self.neighbor_sum(site) + self.h)

            if dE <= 0 or self.rng.random() < np.exp(-dE / self.T):
                self.spins[site] = -s
                self.E += dE
                self.M -= 2 * s


# %%
# Setup checks
model_2d = IsingModel(L=10, dim=2, start="up", seed=0)
model_3d = IsingModel(L=10, dim=3, start="up", seed=0)

assert model_2d.M == model_2d.N
assert model_3d.M == model_3d.N

assert model_2d.E == -2 * model_2d.J * model_2d.N
assert model_3d.E == -3 * model_3d.J * model_3d.N

assert model_2d.neighbor_sum((0, 0)) == 4
assert model_3d.neighbor_sum((0, 0, 0)) == 6

print("2D and 3D setup checks passed.")

# Running E and M should match a fresh recalculation after sweeps.
for dim in (2, 3):
    model = IsingModel(L=10, dim=dim, T=2.0, start="random", seed=1)
    for _ in range(20):
        model.metropolis_sweep()

    assert np.isclose(model.E, model.total_energy())
    assert model.M == model.spins.sum()

# Same check with an external field on.
model = IsingModel(L=10, dim=2, T=2.0, h=0.5, start="random", seed=2)
for _ in range(20):
    model.metropolis_sweep()

assert np.isclose(model.E, model.total_energy())

print("Metropolis bookkeeping checks passed.")


# %%
def sample_magnetization(L, dim, T, h=0.0, n_equil=300, n_measure=300, seed=0):
    """Equilibrate, then return M/N from each measurement sweep."""
    model = IsingModel(L=L, dim=dim, T=T, h=h, start="random", seed=seed)

    for _ in range(n_equil):
        model.metropolis_sweep()

    samples = []
    for _ in range(n_measure):
        model.metropolis_sweep()
        samples.append(model.M / model.N)

    return np.array(samples)


# %%
# Magnetization vs temperature, with and without a field.
temps = np.linspace(1.0, 6.0, 26)
m_2d = [np.mean(np.abs(sample_magnetization(L=8, dim=2, T=T))) for T in temps]
m_3d = [np.mean(np.abs(sample_magnetization(L=8, dim=3, T=T))) for T in temps]

plt.plot(temps, m_2d, "o-", label="2D")
plt.plot(temps, m_3d, "s-", label="3D")
plt.axvline(2.269, ls="--", color="C0", label="2D T_c ≈ 2.269")
plt.axvline(4.511, ls="--", color="C1", label="3D T_c ≈ 4.511")
plt.xlabel("T (units of J/k_B)")
plt.ylabel("<|M|>/N")
plt.legend()
plt.savefig("/Users/jamesmcmillen/ising_magnetization.png", dpi=150)
plt.show()


# %%
# Spin configurations at a few temperatures.
def plot_snapshots(L, dim, temps, n_equil=300, seed=0):
    """Show one equilibrated spin configuration at each temperature."""
    fig, axes = plt.subplots(1, len(temps), figsize=(3 * len(temps), 3))

    for ax, T in zip(axes, temps):
        model = IsingModel(L=L, dim=dim, T=T, start="random", seed=seed)
        for _ in range(n_equil):
            model.metropolis_sweep()

        # In 3D, show the middle slice.
        slice_ = model.spins if dim == 2 else model.spins[:, :, L // 2]
        ax.imshow(slice_, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title(f"T = {T:.2f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plot_snapshots(L=32, dim=2, temps=[1.0, 2.269, 4.0])
plot_snapshots(L=16, dim=3, temps=[2.0, 4.511, 6.0])


# %%
# Magnetization vs external field, below and above T_c (2D).
fields = np.linspace(-0.5, 0.5, 21)

for T in (1.5, 3.0):
    m_vs_h = [np.mean(sample_magnetization(L=8, dim=2, T=T, h=h)) for h in fields]
    plt.plot(fields, m_vs_h, "o-", label=f"T = {T}")

plt.xlabel("h (external field)")
plt.ylabel("<M>/N")
plt.legend()
plt.show()
# %%
# 3D view of the whole lattice: red = spin up, blue = spin down.

def plot_spins_3d(L, temps, n_equil=300, seed=0):
    """Scatter every spin in a 3D lattice at each temperature."""
    fig = plt.figure(figsize=(4 * len(temps), 4))
    x, y, z = np.meshgrid(range(L), range(L), range(L), indexing="ij")

    for i, T in enumerate(temps, start=1):
        model = IsingModel(L=L, dim=3, T=T, start="random", seed=seed)
        for _ in range(n_equil):
            model.metropolis_sweep()

        ax = fig.add_subplot(1, len(temps), i, projection="3d")
        up = model.spins == 1
        ax.scatter(x[up], y[up], z[up], c="tab:red", s=20, depthshade=False)
        ax.scatter(x[~up], y[~up], z[~up], c="tab:blue", s=20, depthshade=False)
        ax.set_title(f"T = {T:.2f}")
        ax.set_axis_off()

    plt.tight_layout()
    plt.show()


plot_spins_3d(L=12, temps=[2.0, 4.511, 6.0])

# %%
# Hysteresis: sweep h up and then back down, starting each step from the
# spins left by the previous step.
def hysteresis_loop(L, dim, T, h_max=0.5, n_steps=21,
                    n_equil=200, n_measure=200, seed=0):
    """Return the field values and the mean M/N at each field."""
    model = IsingModel(L=L, dim=dim, T=T, h=-h_max, start="random", seed=seed)

    up = np.linspace(-h_max, h_max, n_steps)
    fields = np.concatenate([up, up[::-1]])

    mags = []
    for h in fields:
        model.h = h
        model.E = model.total_energy()

        for _ in range(n_equil):
            model.metropolis_sweep()

        samples = []
        for _ in range(n_measure):
            model.metropolis_sweep()
            samples.append(model.M / model.N)

        mags.append(np.mean(samples))

    return fields, np.array(mags)


fields, mags = hysteresis_loop(L=8, dim=2, T=1.5)

plt.plot(fields, mags, "o-")
plt.xlabel("h (external field)")
plt.ylabel("<M>/N")
plt.title("Hysteresis, 2D, T = 1.5")
plt.show()
# %%


# %%
# Interactive 3D view: drag to rotate, scroll to zoom, hover to read a spin's position.
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_spins_3d_interactive(L, temps, n_equil=300, seed=0, html_path=None):
    """Interactive 3D spin plot at each temperature (red = up, blue = down)."""
    fig = make_subplots(
        rows=1, cols=len(temps),
        specs=[[{"type": "scene"} for _ in temps]],
        subplot_titles=[f"T = {T:.2f}" for T in temps],
    )
    x, y, z = np.meshgrid(range(L), range(L), range(L), indexing="ij")

    for col, T in enumerate(temps, start=1):
        model = IsingModel(L=L, dim=3, T=T, start="random", seed=seed)
        for _ in range(n_equil):
            model.metropolis_sweep()

        up = model.spins == 1
        for mask, name, color in ((up, "spin up", "red"), (~up, "spin down", "blue")):
            fig.add_trace(
                go.Scatter3d(
                    x=x[mask], y=y[mask], z=z[mask],
                    mode="markers",
                    marker=dict(size=4, color=color),
                    name=name,
                    showlegend=(col == 1),
                ),
                row=1, col=col,
            )

    fig.update_layout(height=500, width=350 * len(temps))

    if html_path:
        fig.write_html(html_path)
    fig.show()
    return fig


plot_spins_3d_interactive(
    L=12, temps=[2.0, 4.511, 6.0],
    html_path="/Users/jamesmcmillen/ising_3d.html",
)


# %%
# More accurate temperature scan: bigger lattices, more sweeps, error bars,
# and extra temperatures close to each critical point.
def scan_abs_magnetization(L, dim, temps, h=0.0, n_equil=500, n_measure=500,
                           n_blocks=10, seed=0):
    """Return mean |M|/N and its standard error at each temperature."""
    means, errs = [], []

    for T in temps:
        samples = np.abs(sample_magnetization(
            L=L, dim=dim, T=T, h=h, n_equil=n_equil, n_measure=n_measure, seed=seed,
        ))

        # Average within blocks so neighboring sweeps don't count as independent.
        per_block = len(samples) // n_blocks
        blocks = samples[:n_blocks * per_block].reshape(n_blocks, per_block).mean(axis=1)

        means.append(samples.mean())
        errs.append(blocks.std(ddof=1) / np.sqrt(n_blocks))

    return np.array(means), np.array(errs)


def temperature_grid(tc):
    """Coarse grid across the range, plus extra points near tc."""
    coarse = np.linspace(1.0, 6.0, 11)
    near = np.linspace(tc - 0.5, tc + 0.5, 11)
    return np.unique(np.round(np.concatenate([coarse, near]), 3))


scans = {}
for dim, L, tc in ((2, 16, 2.269), (3, 10, 4.511)):
    temps = temperature_grid(tc)
    means, errs = scan_abs_magnetization(L=L, dim=dim, temps=temps)
    scans[dim] = (L, tc, temps, means, errs)

for dim, (L, tc, temps, means, errs) in scans.items():
    plt.errorbar(temps, means, yerr=errs, fmt="o-", capsize=2,
                 label=f"{dim}D, L={L}")
    plt.axvline(tc, ls="--", color="gray")

plt.xlabel("T (units of J/k_B)")
plt.ylabel("<|M|>/N")
plt.title("Magnetization vs temperature, with error bars")
plt.legend()
plt.show()


# %%
# Overlay a small field on the zero-field scan, one panel per dimension.
h_small = 0.05
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, dim in zip(axes, (2, 3)):
    L, tc, temps, means0, errs0 = scans[dim]
    means_h, errs_h = scan_abs_magnetization(L=L, dim=dim, temps=temps, h=h_small)

    ax.errorbar(temps, means0, yerr=errs0, fmt="o-", capsize=2, label="h = 0")
    ax.errorbar(temps, means_h, yerr=errs_h, fmt="s-", capsize=2, label=f"h = {h_small}")
    ax.axvline(tc, ls="--", color="gray")
    ax.set_title(f"{dim}D, L={L}")
    ax.set_xlabel("T (units of J/k_B)")

axes[0].set_ylabel("<|M|>/N")
axes[0].legend()
plt.tight_layout()
plt.show()

# %%
def scan_abs_magnetization(L, dim, temps, h=0.0, n_equil=500, n_measure=500,
                           n_blocks=10, seed=0):
    """Return mean |M|/N and its standard error at each temperature."""
    means, errs = [], []

    for T in temps:
        samples = np.abs(sample_magnetization(
            L=L, dim=dim, T=T, h=h, n_equil=n_equil, n_measure=n_measure, seed=seed,
        ))

        # Average within blocks so neighboring sweeps don't count as independent.
        per_block = len(samples) // n_blocks
        blocks = samples[:n_blocks * per_block].reshape(n_blocks, per_block).mean(axis=1)

        means.append(samples.mean())
        errs.append(blocks.std(ddof=1) / np.sqrt(n_blocks))

    return np.array(means), np.array(errs)
# %%
# Overlay a small field on the zero-field scan, one panel per dimension.
h_small = 0.05
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, dim in zip(axes, (2, 3)):
    L, tc, temps, means0, errs0 = scans[dim]
    means_h, errs_h = scan_abs_magnetization(L=L, dim=dim, temps=temps, h=h_small)

    ax.errorbar(temps, means0, yerr=errs0, fmt="o-", capsize=2, label="h = 0")
    ax.errorbar(temps, means_h, yerr=errs_h, fmt="s-", capsize=2, label=f"h = {h_small}")
    ax.axvline(tc, ls="--", color="gray")
    ax.set_title(f"{dim}D, L={L}")
    ax.set_xlabel("T (units of J/k_B)")

axes[0].set_ylabel("<|M|>/N")
axes[0].legend()
plt.tight_layout()
plt.show()


# %%
# Critical exponent Δσ = β/ν from finite-size scaling at T_c:
# at T = T_c, <|M|>/N scales as L^(-Δσ), so the log-log slope gives -Δσ.


def fit_delta_sigma(dim, tc, sizes, n_equil=1000, n_measure=1000):
    """Return the fitted Δσ and the mean |M|/N at each lattice size."""
    means = []
    for L in sizes:
        samples = np.abs(sample_magnetization(
            L=L, dim=dim, T=tc, n_equil=n_equil, n_measure=n_measure,
        ))
        means.append(samples.mean())

    slope, _ = np.polyfit(np.log(sizes), np.log(means), 1)
    return -slope, np.array(means)


delta_2d, m_2d_tc = fit_delta_sigma(dim=2, tc=2.269, sizes=[8, 16, 24, 32])
print(f"2D Δσ ≈ {delta_2d:.3f}  (exact: 0.125)")

delta_3d, m_3d_tc = fit_delta_sigma(dim=3, tc=4.511, sizes=[6, 8, 10, 12])
print(f"3D Δσ ≈ {delta_3d:.3f}  (known: about 0.518)")

# %%
