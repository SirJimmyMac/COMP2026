"""Baseball pitch physics, optimization, Statcast download, and interactive explorer in one file.

Run one of these:
  python3 baseball_pitch.py explore    interactive simulate / sweep menu
  python3 baseball_pitch.py optimize   optimize every pitch type, write pitch_results.csv
  python3 baseball_pitch.py fetch      download 2025 Statcast data into data/
  python3 baseball_pitch.py skubal     optimize Tarik Skubal's pitches inside his own 2025 ranges
  python3 baseball_pitch.py compare    run gradient, Bayesian, annealing, and genetic search on every pitch type

Sections:
  1. Physics: the differentiable flight model (gravity, drag, Magnus force)
  2. Pitch settings: per-pitch bounds, starting values, and break direction
  3. Reward: strike-zone score plus break
  4. Optimizer: runs the reward for every pitch type
  5. Statcast download: pulls 2025 pitch data with pybaseball
  6. Explorer: type in inputs or sweep one input, with plots
  7. Command line entry point
"""
import os
import sys
import warnings

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

# =============================================================================
# 1. PHYSICS
# Coordinates (SI units): x is horizontal across the plate, y points from the
# pitcher toward home plate, z points up. The ball is released at RELEASE_HEIGHT
# and integrated until it reaches the front of the plate.
# =============================================================================

MASS = 0.145  # kg
RADIUS = 0.0366  # m
AIR_DENSITY = 1.225  # kg/m^3
GRAVITY = 9.81  # m/s^2
DRAG_COEFF = 0.35
AREA = jnp.pi * RADIUS**2

RELEASE_HEIGHT = 1.8  # m
PLATE_DISTANCE = 16.6  # m, release point to front of home plate (approximate)
MPH_TO_MPS = 0.44704

# 1 ms output grid; the ball reaches the plate in roughly 0.45 s
SAVE_TIMES = jnp.linspace(0.0, 0.7, 701)


def spin_vector(spin_rpm, spin_axis_deg):
    """Spin vector in the x-z plane: 0 deg is pure backspin, 90 deg is pure sidespin."""
    omega = spin_rpm * 2.0 * jnp.pi / 60.0
    axis = jnp.deg2rad(spin_axis_deg)
    return omega * jnp.array([jnp.cos(axis), 0.0, jnp.sin(axis)])


def lift_coefficient(spin_param):
    # Empirical baseball fit C_L = 1 / (2 + 1/S), with S = r * omega / v.
    # The small epsilon keeps the gradient finite when spin is zero.
    return 1.0 / (2.0 + 1.0 / (spin_param + 1e-9))


def acceleration(vel, omega_vec):
    speed = jnp.sqrt(jnp.sum(vel**2) + 1e-12)
    spin_param = RADIUS * jnp.sqrt(jnp.sum(omega_vec**2) + 1e-12) / speed

    f_gravity = jnp.array([0.0, 0.0, -MASS * GRAVITY])
    f_drag = -0.5 * AIR_DENSITY * DRAG_COEFF * AREA * speed * vel

    cross = jnp.cross(omega_vec, vel)
    direction = cross / jnp.sqrt(jnp.sum(cross**2) + 1e-12)
    f_magnus = 0.5 * AIR_DENSITY * lift_coefficient(spin_param) * AREA * speed**2 * direction

    return (f_gravity + f_drag + f_magnus) / MASS


def _derivative(t, state, omega_vec):
    vel = state[3:]
    return jnp.concatenate([vel, acceleration(vel, omega_vec)])


def initial_state(speed_mph, horiz_deg, vert_deg):
    speed = speed_mph * MPH_TO_MPS
    horiz = jnp.deg2rad(horiz_deg)
    vert = jnp.deg2rad(vert_deg)
    vel = speed * jnp.array([
        jnp.cos(vert) * jnp.sin(horiz),
        jnp.cos(vert) * jnp.cos(horiz),
        jnp.sin(vert),
    ])
    return jnp.concatenate([jnp.array([0.0, 0.0, RELEASE_HEIGHT]), vel])


def simulate(state0, omega_vec):
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_derivative),
        diffrax.Tsit5(),
        t0=0.0,
        t1=SAVE_TIMES[-1],
        dt0=1e-3,
        y0=state0,
        args=omega_vec,
        saveat=diffrax.SaveAt(ts=SAVE_TIMES),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    return solution.ys


def run_pitch(params):
    """params = (speed_mph, spin_rpm, spin_axis_deg, horiz_deg, vert_deg); returns the trajectory."""
    state0 = initial_state(params[0], params[3], params[4])
    return simulate(state0, spin_vector(params[1], params[2]))


def plate_crossing(states):
    """Linearly interpolate the state at the moment the ball reaches the plate."""
    past_plate = states[:, 1] >= PLATE_DISTANCE
    k = jnp.argmax(past_plate)
    before, after = states[k - 1], states[k]
    frac = (PLATE_DISTANCE - before[1]) / (after[1] - before[1])
    return before + frac * (after - before)


def plate_state(params):
    return plate_crossing(run_pitch(params))


# =============================================================================
# 2. PITCH SETTINGS
# Speed and spin ranges are the 5th-95th percentiles of 2025 Statcast data;
# start values are medians. objective is which break the reward maximizes.
# =============================================================================

PITCHES = {
    "FF": dict(name="Four-seam fastball", objective="vertical", start_axis=1.0,
               speed_range=(90.3, 98.6), spin_range=(2058, 2588), start_speed=94.5, start_spin=2324),
    "SI": dict(name="Sinker", objective="horizontal", start_axis=90.0,
               speed_range=(89.0, 98.0), spin_range=(1917, 2453), start_speed=94.0, start_spin=2189),
    "FC": dict(name="Cutter", objective="horizontal", start_axis=90.0,
               speed_range=(85.0, 94.7), spin_range=(2089, 2750), start_speed=89.7, start_spin=2408),
    "SL": dict(name="Slider", objective="horizontal", start_axis=90.0,
               speed_range=(81.1, 90.6), spin_range=(2049, 2816), start_speed=86.4, start_spin=2432),
    "ST": dict(name="Sweeper", objective="horizontal", start_axis=90.0,
               speed_range=(77.5, 87.0), spin_range=(2177, 3043), start_speed=82.7, start_spin=2601),
    "SV": dict(name="Slurve", objective="horizontal", start_axis=90.0,
               speed_range=(77.9, 86.0), spin_range=(2109, 3053), start_speed=81.9, start_spin=2528),
    "CU": dict(name="Curveball", objective="vertical", start_axis=179.0,
               speed_range=(73.3, 85.1), spin_range=(2113, 3115), start_speed=79.9, start_spin=2588,
               start_vert=3.0),
    "KC": dict(name="Knuckle curve", objective="vertical", start_axis=179.0,
               speed_range=(76.6, 88.2), spin_range=(2066, 2940), start_speed=82.9, start_spin=2500),
    "CS": dict(name="Slow curve", objective="vertical", start_axis=179.0,
               speed_range=(48.8, 75.5), spin_range=(1107, 2785), start_speed=71.9, start_spin=2366,
               start_vert=3.0),
    "CH": dict(name="Changeup", objective="vertical", start_axis=1.0,
               speed_range=(79.0, 91.1), spin_range=(1255, 2266), start_speed=86.3, start_spin=1792),
    "FS": dict(name="Splitter", objective="vertical", start_axis=179.0,
               speed_range=(81.3, 92.7), spin_range=(738, 1892), start_speed=86.3, start_spin=1379),
    "FO": dict(name="Forkball", objective="vertical", start_axis=179.0,
               speed_range=(80.4, 86.1), spin_range=(425, 1398), start_speed=83.1, start_spin=1121),
}


def pitch_bounds(cfg):
    lower = jnp.array([cfg["speed_range"][0], cfg["spin_range"][0], 0.0, -3.0, -5.0])
    upper = jnp.array([cfg["speed_range"][1], cfg["spin_range"][1], 360.0, 3.0, 5.0])
    return lower, upper


def pitch_start_params(cfg):
    return jnp.array([cfg["start_speed"], cfg["start_spin"], cfg["start_axis"], 0.0, cfg.get("start_vert", 0.0)])


def pitch_break_axis(cfg):
    """Index into the plate-crossing state: 0 is horizontal (x), 2 is vertical (z)."""
    return 0 if cfg["objective"] == "horizontal" else 2


# Tarik Skubal, 2025 regular season (Statcast, pitcher id 669373). Ranges are 5th-95th percentiles,
# starts are medians. Splitters (3 pitches) are left out because there is too little data.
SKUBAL_STATS = {
    "FF": dict(speed_range=(95.6, 99.6), spin_range=(2204, 2499), start_speed=97.6, start_spin=2356),
    "SI": dict(speed_range=(95.4, 99.3), spin_range=(2115, 2387), start_speed=97.3, start_spin=2248),
    "CH": dict(speed_range=(85.6, 90.7), spin_range=(1681, 2003), start_speed=88.0, start_spin=1833),
    "SL": dict(speed_range=(87.5, 92.9), spin_range=(2106, 2356), start_speed=90.0, start_spin=2236),
    "CU": dict(speed_range=(78.7, 84.3), spin_range=(2334, 2648), start_speed=81.0, start_spin=2518),
}
SKUBAL_PITCHES = {code: {**PITCHES[code], **stats} for code, stats in SKUBAL_STATS.items()}


# =============================================================================
# 3. REWARD
# Reward = strike score + small weight on break. The break weight is kept low so
# that 20 in of break is worth less than losing the strike.
# =============================================================================

INCH = 0.0254  # m
HALF_WIDTH = 8.5 * INCH + RADIUS  # plate half-width plus ball radius
ZONE_BOTTOM = 1.5 * 0.3048 - RADIUS  # 1.5 ft
ZONE_TOP = 3.5 * 0.3048 + RADIUS  # 3.5 ft
SHARPNESS = 50.0  # per meter; larger gives sharper zone edges
STRIKE_WEIGHT = 1.0
BREAK_WEIGHT = 0.02  # per inch of break


def to_physical(theta, lower, upper):
    return lower + (upper - lower) * jax.nn.sigmoid(theta)


def to_unconstrained(params, lower, upper):
    # Clip so starting values at a bound (e.g. spin axis 0) still have a finite logit
    u = jnp.clip((params - lower) / (upper - lower), 1e-4, 1.0 - 1e-4)
    return jnp.log(u / (1.0 - u))


def strike_score(crossing):
    """Smooth 0-1 score that is near 1 when the plate crossing is inside the zone."""
    x, z = crossing[0], crossing[2]
    sig = jax.nn.sigmoid
    return (
        sig(SHARPNESS * (HALF_WIDTH - jnp.abs(x)))
        * sig(SHARPNESS * (z - ZONE_BOTTOM))
        * sig(SHARPNESS * (ZONE_TOP - z))
    )


def in_zone(crossing):
    x, z = crossing[0], crossing[2]
    return (jnp.abs(x) <= HALF_WIDTH) & (z >= ZONE_BOTTOM) & (z <= ZONE_TOP)


def reference_path(params, n=50):
    """Reference path with no spin and no drag: a straight line along the release velocity, with gravity
    pulling it down (the same convention MLB uses for vertical break). Returns n rows of (x, y, z)."""
    state0 = initial_state(params[0], params[3], params[4])
    t = jnp.linspace(0.0, (PLATE_DISTANCE - state0[1]) / state0[4], n)
    x = state0[0] + state0[3] * t
    y = state0[1] + state0[4] * t
    z = state0[2] + state0[5] * t - 0.5 * GRAVITY * t**2
    return jnp.stack([x, y, z], axis=1)


def reference_at_plate(params):
    """(x, z) of the reference path where it reaches the batter's plane (the plate)."""
    last = reference_path(params, n=2)[-1]
    return last[0], last[2]


def break_inches(params, axis):
    """Break in the batter's plane along one axis (0 = horizontal, 2 = vertical), from the reference path.
    Spin and drag are what remain in the break."""
    crossing = plate_state(params)
    x_ref, z_ref = reference_at_plate(params)
    ref_value = x_ref if axis == 0 else z_ref
    return jnp.abs(crossing[axis] - ref_value) / INCH


def reward(params, axis):
    strike = strike_score(plate_state(params))
    return STRIKE_WEIGHT * strike + BREAK_WEIGHT * break_inches(params, axis)


# =============================================================================
# 4. OPTIMIZER
# Adam on unconstrained values, mapped to each pitch's bounds. Runs every pitch type
# and reports break, strike, and how often nearby release conditions still strike.
# =============================================================================

OPT_STEPS = 400
LEARNING_RATE = 0.05
RESULTS_PATH = "pitch_results.csv"


def optimize_pitch(cfg):
    lower, upper = pitch_bounds(cfg)
    axis = pitch_break_axis(cfg)

    def loss(theta):
        return -reward(to_physical(theta, lower, upper), axis)

    loss_and_grad = jax.jit(jax.value_and_grad(loss))
    theta = to_unconstrained(pitch_start_params(cfg), lower, upper)
    opt = optax.adam(LEARNING_RATE)
    opt_state = opt.init(theta)
    for _ in range(OPT_STEPS):
        _, grad = loss_and_grad(theta)
        updates, opt_state = opt.update(grad, opt_state)
        theta = optax.apply_updates(theta, updates)
    return to_physical(theta, lower, upper)


def robustness(params, n=200, seed=0):
    """Fraction of nearby pitches (small release-condition noise) that are still strikes."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, [1.0, 0.0, 0.0, 0.5, 0.5], size=(n, 5))
    samples = jnp.asarray(params) + jnp.asarray(noise)
    crossings = jax.vmap(plate_state)(samples)
    return float(jnp.mean(jax.vmap(in_zone)(crossings)))


def optimize_all(pitches=PITCHES, out_path=RESULTS_PATH):
    rows = []
    for code, cfg in pitches.items():
        params = optimize_pitch(cfg)
        crossing = plate_state(params)
        row = {
            "code": code,
            "pitch": cfg["name"],
            "objective": cfg["objective"],
            "speed_mph": round(float(params[0]), 1),
            "spin_rpm": round(float(params[1])),
            "spin_axis_deg": round(float(params[2]), 1),
            "break_in": round(float(break_inches(params, pitch_break_axis(cfg))), 1),
            "in_zone": bool(in_zone(crossing)),
            "strike_rate_noise": round(robustness(params), 3),
        }
        rows.append(row)
        print(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nSaved {out_path}")


def optimize_skubal():
    """Optimize each of Skubal's pitches inside his own 2025 speed and spin ranges."""
    optimize_all(SKUBAL_PITCHES, out_path="skubal_results.csv")


# =============================================================================
# 5. STATCAST DOWNLOAD
# Pulls the 2025 regular season one month at a time and combines the files.
# =============================================================================

DATA_DIR = "data"
STATCAST_COLUMNS = [
    "game_date", "pitch_type", "pitch_name", "release_speed", "release_spin_rate",
    "spin_axis", "release_pos_x", "release_pos_y", "release_pos_z", "release_extension",
    "vx0", "vy0", "vz0", "ax", "ay", "az", "pfx_x", "pfx_z", "plate_x", "plate_z",
    "zone", "description", "type",
]
STATCAST_MONTHS = [
    ("2025-03-27", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-28"),
]


def fetch_statcast():
    import pybaseball  # imported here so the other modes don't need it

    warnings.filterwarnings("ignore")
    pybaseball.cache.enable()
    os.makedirs(DATA_DIR, exist_ok=True)
    for start, end in STATCAST_MONTHS:
        path = os.path.join(DATA_DIR, f"statcast_{start[:7]}.csv")
        if os.path.exists(path):
            print(f"Skipping {path} (already downloaded)")
            continue
        df = pybaseball.statcast(start, end)
        keep = [c for c in STATCAST_COLUMNS if c in df.columns]
        df[keep].to_csv(path, index=False)
        print(f"{start} to {end}: {len(df)} pitches -> {path}")

    frames = [pd.read_csv(os.path.join(DATA_DIR, f)) for f in sorted(os.listdir(DATA_DIR)) if f.startswith("statcast_")]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(os.path.join(DATA_DIR, "statcast_2025.csv"), index=False)
    print(f"Combined: {len(combined)} pitches")
    print(combined["pitch_name"].value_counts(dropna=False).to_string())


# =============================================================================
# 6. EXPLORER
# Simulate one pitch from typed-in inputs, or sweep one input across a min and max.
# =============================================================================

# name, label, default min, default max
EXPLORER_FIELDS = [
    ("speed_mph", "Speed (mph)", 40.0, 110.0),
    ("spin_rpm", "Spin (rpm)", 0.0, 4000.0),
    ("spin_axis_deg", "Spin axis (deg)", 0.0, 360.0),
    ("horiz_deg", "Horizontal release angle (deg)", -10.0, 10.0),
    ("vert_deg", "Vertical release angle (deg)", -10.0, 10.0),
]
DEFAULT_CFG = PITCHES["ST"]
SWEEP_POINTS = 41


@jax.jit
def metrics(params):
    """[plate x (in), plate z (m), horizontal break (in), vertical break (in), in zone (1 or 0)].
    Breaks are signed and measured from the straight line from release to the plate."""
    crossing = plate_state(params)
    x_ref, z_ref = reference_at_plate(params)
    return jnp.array([
        crossing[0] / INCH,
        crossing[2],
        (crossing[0] - x_ref) / INCH,
        (crossing[2] - z_ref) / INCH,
        in_zone(crossing).astype(jnp.float32),
    ])


def ask(prompt, default):
    text = input(f"{prompt} [{default:g}]: ").strip()
    return float(text) if text else float(default)


def choose_pitch():
    code = input("Pitch code (FF, SI, FC, SL, ST, SV, CU, KC, CS, CH, FS, FO) or blank for a sweeper: ").strip().upper()
    if not code:
        return DEFAULT_CFG
    if code not in PITCHES:
        print(f"Unknown code {code!r}; using the sweeper defaults.")
        return DEFAULT_CFG
    return PITCHES[code]


def print_metrics(params):
    m = np.asarray(metrics(params))
    print(f"  plate x {m[0]:+.1f} in, plate z {m[1]:.2f} m, in zone: {bool(m[4])}")
    print(f"  horizontal break {m[2]:+.1f} in, vertical break {m[3]:+.1f} in")


def plot_pitch(params, path="last_pitch.png"):
    states = np.asarray(run_pitch(params))
    ref = np.asarray(reference_path(params))
    fig, (side, front) = plt.subplots(1, 2, figsize=(11, 4.5))

    arrived = states[:, 1] <= PLATE_DISTANCE
    side.plot(states[arrived, 1], states[arrived, 2], "-", label="Pitch")
    front.plot(states[arrived, 0] / INCH, states[arrived, 2], "-", label="Pitch")
    side.plot(ref[:, 1], ref[:, 2], "--", label="Reference (no spin, no drag)")
    front.plot(ref[:, 0] / INCH, ref[:, 2], "--", label="Reference (no spin, no drag)")

    side.axhspan(ZONE_BOTTOM, ZONE_TOP, color="grey", alpha=0.15)
    side.set_xlabel("Distance to plate (m)")
    side.set_ylabel("Height z (m)")
    side.set_title("Side view")
    side.legend()

    front.add_patch(plt.Rectangle(
        (-HALF_WIDTH / INCH, ZONE_BOTTOM), 2 * HALF_WIDTH / INCH, ZONE_TOP - ZONE_BOTTOM,
        fill=False, edgecolor="black", linestyle=":",
    ))
    front.set_xlabel("Horizontal position (in)")
    front.set_ylabel("Height z (m)")
    front.set_title("Front view (dotted box = strike zone)")
    front.set_xlim(-30, 30)
    front.set_ylim(0.0, 2.2)
    front.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  Saved {path}")
    plt.show()
    plt.close(fig)


def read_pitch_inputs():
    cfg = choose_pitch()
    defaults = np.asarray(pitch_start_params(cfg))
    values = [ask(label, default) for (_, label, _, _), default in zip(EXPLORER_FIELDS, defaults)]
    return jnp.asarray(values, dtype=jnp.float32)


def simulate_once():
    params = read_pitch_inputs()
    print_metrics(params)
    plot_pitch(params)


def animate_pitch(params, path="pitch_animation.html", frames=120):
    """Build an interactive 3D pitch animation with play and pause. Drag to rotate, scroll to zoom.
    Saved as HTML and opened in the browser."""
    import webbrowser
    import plotly.graph_objects as go

    states = np.asarray(run_pitch(params))
    pitch_path = states[states[:, 1] <= PLATE_DISTANCE]
    ref = np.asarray(reference_path(params))
    steps = np.linspace(0, len(pitch_path) - 1, frames).astype(int)
    m = np.asarray(metrics(params))

    zone_x = [-HALF_WIDTH, HALF_WIDTH, HALF_WIDTH, -HALF_WIDTH, -HALF_WIDTH]
    zone_y = [PLATE_DISTANCE] * 5
    zone_z = [ZONE_BOTTOM, ZONE_BOTTOM, ZONE_TOP, ZONE_TOP, ZONE_BOTTOM]

    traces = [
        go.Scatter3d(x=ref[:, 0], y=ref[:, 1], z=ref[:, 2], mode="lines",
                     line=dict(color="grey", dash="dash", width=5), name="Reference (no spin, no drag)"),
        go.Scatter3d(x=zone_x, y=zone_y, z=zone_z, mode="lines",
                     line=dict(color="black", width=6), name="Strike zone"),
        go.Scatter3d(x=pitch_path[:1, 0], y=pitch_path[:1, 1], z=pitch_path[:1, 2], mode="lines",
                     line=dict(color="royalblue", width=6), name="Pitch"),
        go.Scatter3d(x=pitch_path[:1, 0], y=pitch_path[:1, 1], z=pitch_path[:1, 2], mode="markers",
                     marker=dict(size=8, color="darkorange"), name="Ball"),
    ]

    frame_list = []
    for k, i in enumerate(steps):
        frame_list.append(go.Frame(
            name=str(k),
            data=[
                go.Scatter3d(x=pitch_path[:i + 1, 0], y=pitch_path[:i + 1, 1], z=pitch_path[:i + 1, 2]),
                go.Scatter3d(x=pitch_path[i:i + 1, 0], y=pitch_path[i:i + 1, 1], z=pitch_path[i:i + 1, 2]),
            ],
            traces=[2, 3],
        ))

    layout = go.Layout(
        title=f"Horizontal break {m[2]:+.1f} in, vertical break {m[3]:+.1f} in, "
              f"{'strike' if m[4] else 'ball'}",
        scene=dict(
            xaxis=dict(title="x across plate (m)", range=[-1.0, 1.0]),
            yaxis=dict(title="distance to plate (m)", range=[0.0, PLATE_DISTANCE + 0.5]),
            zaxis=dict(title="height (m)", range=[0.0, 2.2]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=2.5, z=1),
            camera=dict(eye=dict(x=0.0, y=-1.8, z=0.25)),  # camera on the pitcher's side, facing the zone
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="left", yanchor="top",
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))]),
            ],
        )],
    )

    fig = go.Figure(data=traces, frames=frame_list, layout=layout)
    fig.write_html(path, auto_open=False)
    print(f"  Saved {path}")
    webbrowser.open(f"file://{os.path.abspath(path)}")


def animate_once():
    params = read_pitch_inputs()
    print_metrics(params)
    animate_pitch(params)


def sweep_limits(cfg, index):
    """Speed and spin default to the pitch's data range; the other inputs use the general limits."""
    if index == 0:
        return cfg["speed_range"]
    if index == 1:
        return cfg["spin_range"]
    return EXPLORER_FIELDS[index][2], EXPLORER_FIELDS[index][3]


def plot_sweep(label, values, results, path="last_sweep.png"):
    horiz, vert, plate_x, in_zone_flags = results[:, 2], results[:, 3], results[:, 0], results[:, 4]
    fig, (brk, plate) = plt.subplots(1, 2, figsize=(11, 4.5))

    brk.plot(values, horiz, label="Horizontal break")
    brk.plot(values, vert, label="Vertical break")
    brk.axhline(0, color="grey", linewidth=0.8)
    brk.set_xlabel(label)
    brk.set_ylabel("Break vs no spin (in)")
    brk.set_title("Break")
    brk.legend()

    plate.plot(values, plate_x, color="tab:blue", label="Plate x")
    plate.axhspan(-HALF_WIDTH / INCH, HALF_WIDTH / INCH, color="grey", alpha=0.15, label="Zone width")
    plate.scatter(values[in_zone_flags == 1], plate_x[in_zone_flags == 1], color="tab:green", s=14, label="Strike")
    plate.scatter(values[in_zone_flags == 0], plate_x[in_zone_flags == 0], color="tab:red", s=14, label="Ball")
    plate.set_xlabel(label)
    plate.set_ylabel("Horizontal position at plate (in)")
    plate.set_title("Strike or ball")
    plate.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  Saved {path}")
    plt.show()
    plt.close(fig)


def sweep_once():
    cfg = choose_pitch()
    defaults = np.asarray(pitch_start_params(cfg))

    print("Inputs to sweep:")
    for i, (_, label, _, _) in enumerate(EXPLORER_FIELDS, start=1):
        print(f"  {i}) {label}")
    index = int(input("Pick a number: ").strip()) - 1
    if not 0 <= index < len(EXPLORER_FIELDS):
        print("Not a valid choice.")
        return

    _, label, _, _ = EXPLORER_FIELDS[index]
    lo_default, hi_default = sweep_limits(cfg, index)
    lo = ask(f"Min {label}", lo_default)
    hi = ask(f"Max {label}", hi_default)

    values = np.linspace(lo, hi, SWEEP_POINTS)
    batch = np.tile(defaults, (SWEEP_POINTS, 1))
    batch[:, index] = values
    results = np.asarray(jax.vmap(metrics)(jnp.asarray(batch, dtype=jnp.float32)))

    print(f"\nAt min ({lo:g}):")
    print_metrics(jnp.asarray(batch[0], dtype=jnp.float32))
    print(f"At max ({hi:g}):")
    print_metrics(jnp.asarray(batch[-1], dtype=jnp.float32))
    print(f"Strike rate across the sweep: {results[:, 4].mean():.0%}")
    plot_sweep(label, values, results)


def explore():
    while True:
        choice = input("\n1) Simulate a pitch  2) Sweep one input  3) 3D animation  4) Quit: ").strip()
        if choice == "1":
            simulate_once()
        elif choice == "2":
            sweep_once()
        elif choice == "3":
            animate_once()
        elif choice == "4":
            break
        else:
            print("Enter 1, 2, 3, or 4.")


# =============================================================================
# 4b. METHOD COMPARISON
# Four ways to search the same reward, each limited to the same budget of reward evaluations:
#   Gradient (Adam): follows gradients through the physics model (section 4)
#   Bayesian: fits a Gaussian-process model of the reward and picks promising points (scikit-optimize)
#   Annealing: random moves that are accepted less often as the search cools (SciPy dual_annealing)
#   Genetic: a population that is selected, combined, and mutated (SciPy differential_evolution,
#            an evolutionary algorithm)
# Each method's best pitch is then scored the same way: break, strike, and strike rate under noise.
# =============================================================================

COMPARE_BUDGET = 400
COMPARE_PATH = "method_comparison.csv"


def _reward_fn(cfg):
    axis = pitch_break_axis(cfg)
    return jax.jit(lambda p: reward(p, axis))


def search_gradient(cfg, _budget):
    # Adam uses one gradient per step, so OPT_STEPS steps count as the budget
    return optimize_pitch(cfg), OPT_STEPS


def search_bayesian(cfg, budget):
    from skopt import gp_minimize
    from skopt.space import Real

    lower, upper = pitch_bounds(cfg)
    f = _reward_fn(cfg)
    dims = [Real(float(lo), float(hi)) for lo, hi in zip(lower, upper)]

    def objective(x):
        return -float(f(jnp.asarray(x, dtype=jnp.float32)))

    result = gp_minimize(objective, dims, n_calls=budget, random_state=0)
    return jnp.asarray(result.x, dtype=jnp.float32), len(result.func_vals)


def search_annealing(cfg, budget):
    from scipy.optimize import dual_annealing

    lower, upper = pitch_bounds(cfg)
    f = _reward_fn(cfg)
    bounds = list(zip(map(float, lower), map(float, upper)))

    def objective(x):
        return -float(f(jnp.asarray(x, dtype=jnp.float32)))

    result = dual_annealing(objective, bounds, maxfun=budget, seed=0)
    return jnp.asarray(result.x, dtype=jnp.float32), int(result.nfev)


def search_genetic(cfg, budget):
    from scipy.optimize import differential_evolution

    lower, upper = pitch_bounds(cfg)
    f = _reward_fn(cfg)
    bounds = list(zip(map(float, lower), map(float, upper)))
    popsize = 5
    per_generation = popsize * len(bounds)
    maxiter = budget // per_generation - 1

    def objective(x):
        return -float(f(jnp.asarray(x, dtype=jnp.float32)))

    result = differential_evolution(objective, bounds, popsize=popsize, maxiter=maxiter,
                                    polish=False, seed=0)
    return jnp.asarray(result.x, dtype=jnp.float32), int(result.nfev)


def search_random(cfg, budget):
    """Baseline: sample random pitches within the bounds and keep the best one."""
    lower, upper = pitch_bounds(cfg)
    f = _reward_fn(cfg)
    rng = np.random.default_rng(0)
    samples = rng.uniform(np.asarray(lower), np.asarray(upper), size=(budget, len(lower)))
    scores = [float(f(jnp.asarray(s, dtype=jnp.float32))) for s in samples]
    return jnp.asarray(samples[int(np.argmax(scores))], dtype=jnp.float32), budget


METHODS = {
    "Gradient (Adam)": search_gradient,
    "Bayesian": search_bayesian,
    "Annealing": search_annealing,
    "Genetic": search_genetic,
    "Random (baseline)": search_random,
}


def compare_methods():
    rows = []
    for code, cfg in PITCHES.items():
        axis = pitch_break_axis(cfg)
        f = _reward_fn(cfg)
        for name, search in METHODS.items():
            params, evals = search(cfg, COMPARE_BUDGET)
            crossing = plate_state(params)
            row = {
                "code": code,
                "method": name,
                "reward": round(float(f(params)), 3),
                "speed_mph": round(float(params[0]), 1),
                "spin_rpm": round(float(params[1])),
                "spin_axis_deg": round(float(params[2]), 1),
                "break_in": round(float(break_inches(params, axis)), 1),
                "in_zone": bool(in_zone(crossing)),
                "strike_rate_noise": round(robustness(params), 3),
                "evaluations": evals,
            }
            rows.append(row)
            print(row)

    table = pd.DataFrame(rows)
    table.to_csv(COMPARE_PATH, index=False)
    best = table.loc[table.groupby("code")["reward"].idxmax(), ["code", "method", "reward"]]
    print()
    print(table.to_string(index=False))
    print("\nBest method per pitch (highest reward):")
    print(best.to_string(index=False))
    print(f"\nSaved {COMPARE_PATH}")


# =============================================================================
# 7. COMMAND LINE ENTRY POINT
# =============================================================================

MODES = {
    "explore": explore,
    "optimize": optimize_all,
    "fetch": fetch_statcast,
    "skubal": optimize_skubal,
    "compare": compare_methods,
}

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in MODES:
        print(__doc__)
        sys.exit(1)
    MODES[mode]()
