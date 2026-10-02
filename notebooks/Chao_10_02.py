#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma = 10
rho = 28
beta = 8/3

# %%# Lorenz equations
def lorenz(t, state):
    x, y, z = state
    sigma = 10
    rho = 28
    beta = 8/3
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Integrate from t=0 to t=100
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)
initial_state = [1.0, 1.0, 1.0]

sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval, method='RK45')

# Plot 3D attractor
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz Attractor')
plt.show()

# %%
# Two trajectories, tiny separation
delta = 1e-8
state1 = [1.0, 1.0, 1.0]
state2 = [1.0 + delta, 1.0, 1.0]

sol1 = solve_ivp(lorenz, t_span, state1, t_eval=t_eval, method='RK45')
sol2 = solve_ivp(lorenz, t_span, state2, t_eval=t_eval, method='RK45')

# Separation over time
dx = sol2.y[0] - sol1.y[0]
dy = sol2.y[1] - sol1.y[1]
dz = sol2.y[2] - sol1.y[2]
d = np.sqrt(dx**2 + dy**2 + dz**2)

plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d)
plt.xlabel('t')
plt.ylabel('d(t)')
plt.title('Trajectory separation (log scale)')
plt.show()
# %%
# Take log of separation
log_d = np.log(d)

# Pick the fitting window — adjust these bounds after looking at the plot
t_start = 5
t_end = 25
mask = (t_eval >= t_start) & (t_eval <= t_end)

# Fit a line: log(d) = λ*t + const
coeffs = np.polyfit(t_eval[mask], log_d[mask], 1)
lam = coeffs[0]
fit_line = np.polyval(coeffs, t_eval[mask])

# Plot with fit overlaid
plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d, label='separation d(t)')
plt.semilogy(t_eval[mask], np.exp(fit_line), 'r--', label=f'fit: λ = {lam:.4f}')
plt.xlabel('t')
plt.ylabel('d(t)')
plt.title('Fitting the Lyapunov exponent')
plt.legend()
plt.show()

print(f"Largest Lyapunov exponent λ ≈ {lam:.4f}")

# %%
# Take log of separation
log_d = np.log(d)

# Pick the fitting window — adjust these bounds after looking at the plot
t_start = 10
t_end = 35
mask = (t_eval >= t_start) & (t_eval <= t_end)

# Fit a line: log(d) = λ*t + const
coeffs = np.polyfit(t_eval[mask], log_d[mask], 1)
lam = coeffs[0]
fit_line = np.polyval(coeffs, t_eval[mask])

# Plot with fit overlaid
plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d, label='separation d(t)')
plt.semilogy(t_eval[mask], np.exp(fit_line), 'r--', label=f'fit: λ = {lam:.4f}')
plt.xlabel('t')
plt.ylabel('d(t)')
plt.title('Fitting the Lyapunov exponent')
plt.legend()
plt.show()

print(f"Largest Lyapunov exponent λ ≈ {lam:.4f}")

# %%
from scipy.stats import linregress

# Slide a window and compute R² of fit at each position
window = 1500  # number of time points in window
best_r2 = 0
best_start = 0

for i in range(len(t_eval) - window):
    t_win = t_eval[i:i+window]
    log_d_win = log_d[i:i+window]
    
    if not np.all(np.isfinite(log_d_win)):
        continue
    
    slope, intercept, r, _, _ = linregress(t_win, log_d_win)
    if r**2 > best_r2:
        best_r2 = r**2
        best_start = i
        best_slope = slope

best_end = best_start + window
t_win = t_eval[best_start:best_end]
intercept_best = np.mean(log_d[best_start:best_end]) - best_slope * np.mean(t_win)
fit_line = best_slope * t_win + intercept_best


plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d, label='separation d(t)')
plt.semilogy(t_win, np.exp(fit_line), 'r--', label=f'best fit: λ = {best_slope:.4f}')
plt.xlabel('t')
plt.ylabel('d(t)')
plt.legend()
plt.show()

print(f"λ ≈ {best_slope:.4f}  (R² = {best_r2:.6f}, window t={t_eval[best_start]:.1f} to {t_eval[best_end]:.1f})")

# %%best_r2 = 0
best_result = {}

for window in range(300, 3000, 100):  # try windows from 3 to 30 time units
    for i in range(len(t_eval) - window):
        t_win = t_eval[i:i+window]
        log_d_win = log_d[i:i+window]
        
        if not np.all(np.isfinite(log_d_win)):
            continue
        
        slope, intercept, r, _, _ = linregress(t_win, log_d_win)
        
        if r**2 > best_r2 and slope > 0:  # slope must be positive
            best_r2 = r**2
            best_result = {'slope': slope, 'intercept': intercept, 
                           'start': i, 'end': i+window, 'window': window}

# Plot best result
t_win = t_eval[best_result['start']:best_result['end']]
fit_line = best_result['slope'] * t_win + best_result['intercept']

plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d, label='separation d(t)')
plt.semilogy(t_win, np.exp(fit_line), 'r--', label=f"λ = {best_result['slope']:.4f}")
plt.legend()
plt.show()

print(f"λ ≈ {best_result['slope']:.4f}")
print(f"R² = {best_r2:.6f}")
print(f"Window: t={t_eval[best_result['start']]:.1f} to {t_eval[best_result['end']]:.1f} ({best_result['window']} points)")


# %%
# okay, now change initial conditions and see if we get the same Lyapunov exponent
#change initial conditions of separation to 1e-6
# Two trajectories, tiny separation
delta = 1e-6
state1 = [1.0, 1.0, 1.0]
state2 = [1.0 + delta, 1.0, 1.0]

sol1 = solve_ivp(lorenz, t_span, state1, t_eval=t_eval, method='RK45')
sol2 = solve_ivp(lorenz, t_span, state2, t_eval=t_eval, method='RK45')

# Separation over time
dx = sol2.y[0] - sol1.y[0]
dy = sol2.y[1] - sol1.y[1]
dz = sol2.y[2] - sol1.y[2]
d = np.sqrt(dx**2 + dy**2 + dz**2)

plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d)
plt.xlabel('t')
plt.ylabel('d(t)')
plt.title('Trajectory separation (log scale)')
plt.show()
# %%
#now test the sliding window method again with the new initial conditions
# %%best_r2 = 0
best_result = {}

for window in range(300, 3000, 100):  # try windows from 3 to 30 time units
    for i in range(len(t_eval) - window):
        t_win = t_eval[i:i+window]
        log_d_win = log_d[i:i+window]
        
        if not np.all(np.isfinite(log_d_win)):
            continue
        
        slope, intercept, r, _, _ = linregress(t_win, log_d_win)
        
        if r**2 > best_r2 and slope > 0:  # slope must be positive
            best_r2 = r**2
            best_result = {'slope': slope, 'intercept': intercept, 
                           'start': i, 'end': i+window, 'window': window}

# Plot best result
t_win = t_eval[best_result['start']:best_result['end']]
fit_line = best_result['slope'] * t_win + best_result['intercept']

plt.figure(figsize=(10, 5))
plt.semilogy(t_eval, d, label='separation d(t)')
plt.semilogy(t_win, np.exp(fit_line), 'r--', label=f"λ = {best_result['slope']:.4f}")
plt.legend()
plt.show()

print(f"λ ≈ {best_result['slope']:.4f}")
print(f"R² = {best_r2:.6f}")
print(f"Window: t={t_eval[best_result['start']]:.1f} to {t_eval[best_result['end']]:.1f} ({best_result['window']} points)")

# %%
print("Any non-finite in log_d:", np.any(~np.isfinite(log_d)))
print("Min d:", d.min())
print("Max d:", d.max())
print("best_result:", best_result)

# %%
from joblib import Parallel, delayed

def measure_lambda(ic):
    state1 = ic
    state2 = [ic[0] + 1e-8, ic[1], ic[2]]
    sol1 = solve_ivp(lorenz, t_span, state1, t_eval=t_eval, method='RK45')
    sol2 = solve_ivp(lorenz, t_span, state2, t_eval=t_eval, method='RK45')
    # ... compute d, fit λ, return slope
    return lam

initial_conditions = [[1,1,1], [2,3,4], [-1,5,2], [0,1,20], [5,-3,10]]

results = Parallel(n_jobs=-1)(delayed(measure_lambda)(ic) for ic in initial_conditions)
print(results)

# %%
from joblib import Parallel, delayed

def measure_lambda(ic):
    delta = 1e-8
    state1 = ic
    state2 = [ic[0] + delta, ic[1], ic[2]]
    
    sol1 = solve_ivp(lorenz, t_span, state1, t_eval=t_eval, method='RK45')
    sol2 = solve_ivp(lorenz, t_span, state2, t_eval=t_eval, method='RK45')
    
    dx = sol2.y[0] - sol1.y[0]
    dy = sol2.y[1] - sol1.y[1]
    dz = sol2.y[2] - sol1.y[2]
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    log_d = np.log(d)
    
    # Sliding window to find best fit
    best_r2 = 0
    best_slope = None
    window = 1500
    
    for i in range(len(t_eval) - window):
        t_win = t_eval[i:i+window]
        log_d_win = log_d[i:i+window]
        
        if not np.all(np.isfinite(log_d_win)):
            continue
        
        slope, intercept, r, _, _ = linregress(t_win, log_d_win)
        
        if r**2 > best_r2 and slope > 0:
            best_r2 = r**2
            best_slope = slope
    
    return best_slope

# Many different starting points
initial_conditions = [
    [1.0, 1.0, 1.0],
    [2.0, 3.0, 4.0],
    [-1.0, 5.0, 2.0],
    [0.0, 1.0, 20.0],
    [5.0, -3.0, 10.0],
    [-5.0, 2.0, 15.0],
    [3.0, -1.0, 8.0],
    [0.5, 0.5, 25.0],
]

results = Parallel(n_jobs=-1)(delayed(measure_lambda)(ic) for ic in initial_conditions)
results = [r for r in results if r is not None]

print("λ estimates:", [f"{r:.4f}" for r in results])
print(f"\nMean λ = {np.mean(results):.4f}")
print(f"Std  λ = {np.std(results):.4f}")

# %%
from joblib import Parallel, delayed

def measure_lambda(ic):
    delta = 1e-8
    state1 = ic
    state2 = [ic[0] + delta, ic[1], ic[2]]
    
    sol1 = solve_ivp(lorenz, t_span, state1, t_eval=t_eval, method='RK45')
    sol2 = solve_ivp(lorenz, t_span, state2, t_eval=t_eval, method='RK45')
    
    dx = sol2.y[0] - sol1.y[0]
    dy = sol2.y[1] - sol1.y[1]
    dz = sol2.y[2] - sol1.y[2]
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    log_d = np.log(d)
    
    best_r2 = 0
    best_slope = None
    window = 1500
    
    for i in range(len(t_eval) - window):
        t_win = t_eval[i:i+window]
        log_d_win = log_d[i:i+window]
        
        if not np.all(np.isfinite(log_d_win)):
            continue
        
        slope, intercept, r, _, _ = linregress(t_win, log_d_win)
        
        if r**2 > best_r2 and slope > 0:
            best_r2 = r**2
            best_slope = slope
    
    return best_slope

initial_conditions = [
    [1.0, 1.0, 1.0],
    [2.0, 3.0, 4.0],
    [-1.0, 5.0, 2.0],
    [0.0, 1.0, 20.0],
    [5.0, -3.0, 10.0],
    [-5.0, 2.0, 15.0],
    [3.0, -1.0, 8.0],
    [0.5, 0.5, 25.0],
    [4.0, -4.0, 12.0],
    [-3.0, 6.0, 18.0],
    [2.0, -2.0, 6.0],
    [-2.0, 4.0, 22.0],
]

results = Parallel(n_jobs=-1)(delayed(measure_lambda)(ic) for ic in initial_conditions)
results = [r for r in results if r is not None]

# %%
print("λ estimates:", [f"{r:.4f}" for r in results])
print(f"\nMean λ = {np.mean(results):.4f}")
print(f"Std  λ = {np.std(results):.4f}")
# %%
# Full Lyapunov spectrum via QR decomposition method

sigma, rho, beta = 10, 28, 8/3

def lorenz_variational(t, state_and_phi):
    x, y, z = state_and_phi[:3]
    phi = state_and_phi[3:].reshape(3, 3)
    
    # Lorenz derivatives
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    
    # Jacobian at current point
    J = np.array([[-sigma, sigma,    0],
                  [rho-z,    -1,   -x],
                  [y,         x, -beta]])
    
    # Variational equation: dΦ/dt = J @ Φ
    dphi = J @ phi
    
    return np.concatenate([[dx, dy, dz], dphi.flatten()])

# Initial state + identity matrix as perturbation basis
x0 = [1.0, 1.0, 1.0]
state0 = np.concatenate([x0, np.eye(3).flatten()])

# Integrate in short chunks, re-orthogonalize with QR each time
dt = 0.1
n_steps = 10000
t = 0
state = state0
lyap_sum = np.zeros(3)

for i in range(n_steps):
    sol = solve_ivp(lorenz_variational, [t, t+dt], state, method='RK45')
    state = sol.y[:, -1]
    t += dt
    
    # QR decomposition of perturbation matrix
    phi = state[3:].reshape(3, 3)
    Q, R = np.linalg.qr(phi)
    
    # Accumulate log growth rates
    lyap_sum += np.log(np.abs(np.diag(R)))
    
    # Reset to orthonormal basis
    state[3:] = Q.flatten()

# Average over time
exponents = lyap_sum / (n_steps * dt)
exponents = np.sort(exponents)[::-1]  # descending order

print("Lyapunov spectrum:")
for i, lam in enumerate(exponents):
    print(f"  λ{i+1} = {lam:.4f}")

print(f"\nSum       = {np.sum(exponents):.4f}")
print(f"Expected  = {-(sigma + 1 + beta):.4f}")
print(f"Error     = {abs(np.sum(exponents) - (-(sigma + 1 + beta))):.6f}")

# %%
