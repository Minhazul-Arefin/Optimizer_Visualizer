import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Optimizer Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2.2rem !important; padding-bottom: 0.5rem !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 4px 12px; font-size: 13px; }
    section[data-testid="stSidebar"] { width: 285px !important; }
    .stSlider { margin-bottom: -8px !important; }
    div[data-testid="metric-container"] { padding: 4px 8px !important; }
    h1 { font-size: 1.2rem !important; margin-bottom: 0 !important; }
    h2 { font-size: 1.05rem !important; margin: 4px 0 4px !important; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "SGD": "#e24b4a",
    "Momentum": "#378add",
    "RMSProp": "#1d9e75",
    "AdamW": "#ba7517",
}
EPS = 1e-8

PRESETS = {
    "Custom": None,
    "Ill-conditioned valley": {
        "surface": "Rosenbrock",
        "lr": 0.03,
        "b1": 0.90,
        "b2": 0.99,
        "wd": 0.005,
        "steps": 180,
        "sigma": 0.0,
        "batchsz": 32,
        "perturb": 0.02,
        "log_y": True,
        "note": "Classic narrow curved valley. Useful for showing why plain SGD zig-zags while momentum and adaptive methods stabilize progress.",
    },
    "Noisy training": {
        "surface": "Rosenbrock",
        "lr": 0.05,
        "b1": 0.90,
        "b2": 0.99,
        "wd": 0.01,
        "steps": 160,
        "sigma": 0.60,
        "batchsz": 4,
        "perturb": 0.02,
        "log_y": True,
        "note": "High stochasticity regime. Good for comparing how SGD, RMSProp, and AdamW react when gradient estimates are noisy.",
    },
    "Saddle escape": {
        "surface": "Saddle valley",
        "lr": 0.06,
        "b1": 0.92,
        "b2": 0.99,
        "wd": 0.002,
        "steps": 140,
        "sigma": 0.05,
        "batchsz": 16,
        "perturb": 0.03,
        "log_y": False,
        "note": "Mixed-curvature region. Use this to highlight how momentum and adaptive methods move through flat but unstable directions.",
    },
    "Multi-modal landscape": {
        "surface": "Rastrigin",
        "lr": 0.04,
        "b1": 0.90,
        "b2": 0.99,
        "wd": 0.005,
        "steps": 220,
        "sigma": 0.02,
        "batchsz": 16,
        "perturb": 0.02,
        "log_y": True,
        "note": "Highly multi-modal landscape with many local traps. Good for showing optimizer sensitivity to local geometry.",
    },
}

# ─────────────────────────────────────────────────────────────
# Loss surfaces
# ─────────────────────────────────────────────────────────────
def rosenbrock(x, y):
    return np.log1p((1 - x) ** 2 + 100 * (y - x ** 2) ** 2) * 0.25

def beale(x, y):
    return np.log1p(
        (1.5 - x + x * y) ** 2
        + (2.25 - x + x * y ** 2) ** 2
        + (2.625 - x + x * y ** 3) ** 2
    ) * 0.12

def saddle_v(x, y):
    return 0.5 * (np.tanh(x ** 2 * 0.4 - y ** 2 * 0.5 + 0.2 * np.sin(2.5 * x) * np.cos(1.8 * y)) + 1)

def himmelblau(x, y):
    return np.log1p((x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2) * 0.18

def rastrigin(x, y):
    return np.clip((x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y) + 20) / 80, 0, 1)

def ackley(x, y):
    a = -0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))
    b = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return np.clip((np.e + 20 - 20 * np.exp(a) - np.exp(b)) / 22, 0, 1)

def styblinski_tang(x, y):
    raw = 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x + y ** 4 - 16 * y ** 2 + 5 * y)
    return np.clip((raw + 160) / 320, 0, 1)

def eggholder_scaled(x, y):
    X, Y = 128 * x, 128 * y
    raw = -(Y + 47) * np.sin(np.sqrt(np.abs(X / 2 + (Y + 47)))) - X * np.sin(np.sqrt(np.abs(X - (Y + 47))))
    return np.clip((raw + 1000) / 2000, 0, 1)

def six_hump_camel(x, y):
    raw = (4 - 2.1 * x ** 2 + x ** 4 / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2
    return np.clip((raw + 2) / 6, 0, 1)

def levy(x, y):
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4
    raw = (
        np.sin(np.pi * w1) ** 2
        + (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
        + (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)
    )
    return np.clip(raw / 30, 0, 1)

SURFACES = {
    "Rosenbrock":      {"f": rosenbrock,      "xr": (-2, 2),    "yr": (-0.5, 3),   "start": (-1.5, 1.0),  "h_fd": 1e-4},
    "Beale":           {"f": beale,           "xr": (-4, 4),    "yr": (-4, 4),     "start": (-3.0, 2.5),  "h_fd": 4e-4},
    "Saddle valley":   {"f": saddle_v,        "xr": (-3, 3),    "yr": (-3, 3),     "start": (-2.0, 2.0),  "h_fd": 3e-4},
    "Himmelblau":      {"f": himmelblau,      "xr": (-5, 5),    "yr": (-5, 5),     "start": (-4.0, 4.0),  "h_fd": 5e-4},
    "Rastrigin":       {"f": rastrigin,       "xr": (-4, 4),    "yr": (-4, 4),     "start": (-3.5, 3.5),  "h_fd": 4e-4},
    "Ackley":          {"f": ackley,          "xr": (-4, 4),    "yr": (-4, 4),     "start": (-3.0, 2.5),  "h_fd": 4e-4},
    "Styblinski-Tang": {"f": styblinski_tang, "xr": (-5, 5),    "yr": (-5, 5),     "start": (-4.0, 3.5),  "h_fd": 5e-4},
    "Eggholder":       {"f": eggholder_scaled,"xr": (-4, 4),    "yr": (-4, 4),     "start": (-3.2, 2.4),  "h_fd": 4e-4},
    "Six-Hump Camel":  {"f": six_hump_camel,  "xr": (-2, 2),    "yr": (-1.2, 1.2), "start": (-1.5, 0.8),  "h_fd": 2e-4},
    "Levy":            {"f": levy,            "xr": (-5, 5),    "yr": (-5, 5),     "start": (-4.0, 3.5),  "h_fd": 5e-4},
}

# ─────────────────────────────────────────────────────────────
# Numerical helpers
# ─────────────────────────────────────────────────────────────
def grad(f, x, y, h):
    return (f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)

def hessian_eigs(f, x, y, h):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h ** 2
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h ** 2
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    H = np.array([[fxx, fxy], [fxy, fyy]])
    evals, evecs = np.linalg.eigh(H)
    l2, l1 = float(evals[0]), float(evals[1])
    return l1, l2, fxx, fyy, fxy, evecs

@st.cache_data(show_spinner=False)
def make_grid(xr, yr, res, surface_name):
    surf = SURFACES[surface_name]
    X = np.linspace(*xr, res)
    Y = np.linspace(*yr, res)
    Xg, Yg = np.meshgrid(X, Y)
    Z = np.clip(surf["f"](Xg, Yg), 0, 1)
    return Xg, Yg, Z

# ─────────────────────────────────────────────────────────────
# Optimizer engine + instrumentation
# ─────────────────────────────────────────────────────────────
def make_states(start):
    sx, sy = start
    return {
        "SGD": {"x": sx, "y": sy},
        "Momentum": {"x": sx, "y": sy, "vx": 0.0, "vy": 0.0},
        "RMSProp": {"x": sx, "y": sy, "vx": EPS, "vy": EPS},
        "AdamW": {"x": sx, "y": sy, "mx": 0.0, "my": 0.0, "vx": EPS, "vy": EPS, "t": 0},
    }

def _append_internal(record, name, gx, gy, step_x, step_y, state_dict):
    gnorm = float(np.sqrt(gx ** 2 + gy ** 2))
    snorm = float(np.sqrt(step_x ** 2 + step_y ** 2))
    vnorm = float(np.sqrt(state_dict.get("vx", 0.0) ** 2 + state_dict.get("vy", 0.0) ** 2))
    mnorm = float(np.sqrt(state_dict.get("mx", 0.0) ** 2 + state_dict.get("my", 0.0) ** 2))
    record[name]["grad_norm"].append(gnorm)
    record[name]["step_norm"].append(snorm)
    record[name]["velocity_norm"].append(vnorm)
    record[name]["moment_norm"].append(mnorm)
    record[name]["eff_lr"].append(snorm / (gnorm + EPS))

    if name == "RMSProp":
        record[name]["second_moment_norm"].append(vnorm)
    elif name == "AdamW":
        second_norm = float(np.sqrt(state_dict["vx"] ** 2 + state_dict["vy"] ** 2))
        record[name]["second_moment_norm"].append(second_norm)
    else:
        record[name]["second_moment_norm"].append(0.0)

def step_all(states, f, lr, b1, b2, wd, h, internal_history, noise=0.0):
    for name, o in states.items():
        gx, gy = grad(f, o["x"], o["y"], h)
        if noise > 0:
            gx += noise * np.random.randn()
            gy += noise * np.random.randn()

        prev_x, prev_y = o["x"], o["y"]

        if name == "SGD":
            o["x"] -= lr * gx
            o["y"] -= lr * gy

        elif name == "Momentum":
            o["vx"] = b1 * o["vx"] + gx
            o["vy"] = b1 * o["vy"] + gy
            o["x"] -= lr * o["vx"]
            o["y"] -= lr * o["vy"]

        elif name == "RMSProp":
            o["vx"] = b2 * o["vx"] + (1 - b2) * gx ** 2
            o["vy"] = b2 * o["vy"] + (1 - b2) * gy ** 2
            o["x"] -= lr * gx / (np.sqrt(o["vx"]) + EPS)
            o["y"] -= lr * gy / (np.sqrt(o["vy"]) + EPS)

        elif name == "AdamW":
            o["t"] += 1
            o["mx"] = b1 * o["mx"] + (1 - b1) * gx
            o["my"] = b1 * o["my"] + (1 - b1) * gy
            o["vx"] = b2 * o["vx"] + (1 - b2) * gx ** 2
            o["vy"] = b2 * o["vy"] + (1 - b2) * gy ** 2
            mxh = o["mx"] / (1 - b1 ** o["t"])
            myh = o["my"] / (1 - b1 ** o["t"])
            vxh = o["vx"] / (1 - b2 ** o["t"])
            vyh = o["vy"] / (1 - b2 ** o["t"])
            o["x"] = o["x"] * (1 - lr * wd) - lr * mxh / (np.sqrt(vxh) + EPS)
            o["y"] = o["y"] * (1 - lr * wd) - lr * myh / (np.sqrt(vyh) + EPS)

        step_x = o["x"] - prev_x
        step_y = o["y"] - prev_y
        _append_internal(internal_history, name, gx, gy, step_x, step_y, o)

def run_sim(f, start, n, lr, b1, b2, wd, h, noise=0.0):
    states = make_states(start)
    paths = {k: [(states[k]["x"], states[k]["y"])] for k in states}
    losses = {k: [] for k in states}
    internal_history = {
        k: {
            "grad_norm": [],
            "step_norm": [],
            "velocity_norm": [],
            "moment_norm": [],
            "second_moment_norm": [],
            "eff_lr": [],
        }
        for k in states
    }

    for _ in range(n):
        step_all(states, f, lr, b1, b2, wd, h, internal_history, noise)
        for k in states:
            paths[k].append((states[k]["x"], states[k]["y"]))
            losses[k].append(float(f(states[k]["x"], states[k]["y"])))

    return paths, losses, internal_history, states

FW, FH = 6.2, 4.0

# ─────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    preset_name = st.selectbox("Preset experiment", list(PRESETS.keys()))
    preset_cfg = PRESETS[preset_name]
    using_preset = preset_cfg is not None

    default_surface = preset_cfg["surface"] if using_preset else list(SURFACES.keys())[0]
    surf_name = st.selectbox("Loss surface", list(SURFACES.keys()), index=list(SURFACES.keys()).index(default_surface))
    surf_cfg = SURFACES[surf_name]

    st.markdown("**Optimizer hyperparameters**")
    lr = st.slider("Learning rate α", 0.01, 0.20, preset_cfg["lr"] if using_preset else 0.05, 0.01)
    b1 = st.slider("β₁ (momentum)", 0.50, 0.99, preset_cfg["b1"] if using_preset else 0.90, 0.01)
    b2 = st.slider("β₂ (RMS/Adam)", 0.80, 0.999, preset_cfg["b2"] if using_preset else 0.99, 0.001, format="%.3f")
    wd = st.slider("Weight decay λ", 0.0, 0.10, preset_cfg["wd"] if using_preset else 0.01, 0.005)
    n = st.slider("Steps", 20, 500, preset_cfg["steps"] if using_preset else 150, 10)

    st.markdown("---")
    st.markdown("**Hessian probe**")
    x0h, x1h = surf_cfg["xr"]
    y0h, y1h = surf_cfg["yr"]
    default_probe_x = float(surf_cfg["start"][0])
    default_probe_y = float(surf_cfg["start"][1])
    hpx = st.slider("Probe x", float(x0h), float(x1h), default_probe_x, 0.05)
    hpy = st.slider("Probe y", float(y0h), float(y1h), default_probe_y, 0.05)

    st.markdown("**Stochastic noise**")
    sigma = st.slider("Noise σ", 0.0, 1.0, preset_cfg["sigma"] if using_preset else 0.30, 0.05)
    batchsz = st.slider("Batch size B", 1, 64, preset_cfg["batchsz"] if using_preset else 8, 1)

    st.markdown("**Saddle escape**")
    perturb = st.slider("Perturbation", 0.001, 0.2, preset_cfg["perturb"] if using_preset else 0.02, 0.005)

    st.markdown("**Convergence**")
    log_y = st.checkbox("Log-scale y", value=preset_cfg["log_y"] if using_preset else True)

    st.markdown("---")
    st.markdown("**Optimizer internals**")
    selected_step = st.slider("Inspect iteration", 1, n, min(20, n), 1)

    if using_preset:
        st.success(f"Preset active: {preset_name}")
        st.caption(preset_cfg["note"])

    st.caption("@Minhazul Arefin")

st.markdown("# Optimizer Visualizer")
st.caption("SGD · Momentum · RMSProp · AdamW  |  Includes optimizer internals and preset experiments for reproducible comparisons.")
if preset_name != "Custom":
    st.info(f"Running preset **{preset_name}** on **{surf_name}**.")

# Shared main simulation
paths, losses, internals, final_states = run_sim(
    surf_cfg["f"], surf_cfg["start"], n, lr, b1, b2, wd, surf_cfg["h_fd"]
)

(tab1, tab2, tab3, tab4, tab5, tab6) = st.tabs([
    "🌐 3D Surface",
    "📐 Hessian",
    "🎲 Stochastic",
    "⚡ Saddle Escape",
    "📈 Convergence",
    "🧠 Optimizer Internals",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — 3D surface race
# ─────────────────────────────────────────────────────────────
with tab1:
    left, right = st.columns([1, 2.4], gap="small")

    with left:
        st.markdown("**What you're seeing**")
        st.caption("Drag to rotate · Scroll to zoom · Click legend to toggle optimizers. Tune LR, β₁, β₂, and steps in the sidebar.")
        st.markdown("**Final loss**")
        c1, c2 = st.columns(2)
        for i, name in enumerate(COLORS):
            (c1 if i % 2 == 0 else c2).metric(name, f"{losses[name][-1]:.5f}")

    with right:
        Xg, Yg, Z = make_grid(surf_cfg["xr"], surf_cfg["yr"], 80, surf_name)
        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            x=Xg, y=Yg, z=Z,
            colorscale="Blues",
            reversescale=True,
            opacity=0.72,
            showscale=False,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=False)),
            name="Loss surface",
            hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>f: %{z:.4f}<extra>Surface</extra>",
        ))
        for name, path in paths.items():
            xs, ys = zip(*path)
            zs = [float(np.clip(surf_cfg["f"](x, y), 0, 1)) + 0.03 for x, y in path]
            fig3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines+markers",
                name=name,
                line=dict(color=COLORS[name], width=5),
                marker=dict(size=[3] * len(xs), color=COLORS[name]),
                hovertemplate=f"<b>{name}</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>f: %{{z:.4f}}<extra></extra>",
            ))
            fig3d.add_trace(go.Scatter3d(
                x=[xs[-1]], y=[ys[-1]], z=[zs[-1]],
                mode="markers",
                name=f"{name} (end)",
                marker=dict(size=8, color=COLORS[name], symbol="circle", line=dict(color="white", width=1)),
                showlegend=False,
                hovertemplate=f"<b>{name} final</b><br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>f: %{{z:.4f}}<extra></extra>",
            ))
        fig3d.update_layout(
            height=480,
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(
                xaxis_title="x", yaxis_title="y", zaxis_title="f(x,y)",
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=0.6),
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(text=f"<b>{surf_name}</b> — drag to rotate, scroll to zoom", font=dict(size=12), x=0.5),
        )
        st.plotly_chart(fig3d, use_container_width=True, config={"displayModeBar": True})

# ─────────────────────────────────────────────────────────────
# TAB 2 — Gradient + Hessian
# ─────────────────────────────────────────────────────────────
with tab2:
    f_h = surf_cfg["f"]
    h_fd = surf_cfg["h_fd"]
    gx_pt, gy_pt = grad(f_h, hpx, hpy, h_fd)
    l1, l2, _, _, _, evecs = hessian_eigs(f_h, hpx, hpy, h_fd * 10)
    gmag = np.sqrt(gx_pt ** 2 + gy_pt ** 2) + EPS
    curve_type = (
        "✅ Convex — local minimum" if l1 > 0 and l2 > 0 else
        "❌ Concave — local maximum" if l1 < 0 and l2 < 0 else
        "⚠️ Saddle point"
    )

    left, right = st.columns([1, 2.4], gap="small")
    with left:
        st.markdown("**Gradient at probe**")
        c1, c2 = st.columns(2)
        c1.metric("∂f/∂x", f"{gx_pt:.4f}")
        c2.metric("∂f/∂y", f"{gy_pt:.4f}")
        st.metric("|∇f|", f"{gmag:.4f}")
        st.markdown("**Hessian eigenvalues**")
        c1, c2 = st.columns(2)
        c1.metric("λ₁", f"{l1:.4f}")
        c2.metric("λ₂", f"{l2:.4f}")
        st.info(curve_type)
        st.caption(f"Finite-difference step: gradient h={h_fd:.1e}, Hessian h={h_fd * 10:.1e}")

    with right:
        Xg, Yg, Z = make_grid(surf_cfg["xr"], surf_cfg["yr"], 150, surf_name)
        fig2 = plt.figure(figsize=(FW * 1.18, FH))
        ax2 = fig2.add_axes([0.07, 0.12, 0.62, 0.78])
        ax2.contourf(Xg, Yg, Z, levels=40, cmap="Blues_r", alpha=0.85)
        ax2.contour(Xg, Yg, Z, levels=20, colors="white", linewidths=0.3, alpha=0.35)
        xs_g = np.linspace(*surf_cfg["xr"], 11)
        ys_g = np.linspace(*surf_cfg["yr"], 11)
        for xi in xs_g:
            for yi in ys_g:
                gxi, gyi = grad(f_h, xi, yi, h_fd)
                mg = np.sqrt(gxi ** 2 + gyi ** 2) + EPS
                sc = min(1.0, mg / 3) * 0.14 * (surf_cfg["xr"][1] - surf_cfg["xr"][0]) / 8
                ax2.annotate("", xy=(xi - gxi / mg * sc, yi - gyi / mg * sc), xytext=(xi, yi),
                             arrowprops=dict(arrowstyle="-|>", color="white", lw=0.6, alpha=0.4))
        al = (surf_cfg["xr"][1] - surf_cfg["xr"][0]) * 0.10
        ax2.annotate("", xy=(hpx - gx_pt / gmag * al, hpy - gy_pt / gmag * al), xytext=(hpx, hpy),
                     arrowprops=dict(arrowstyle="-|>", color="#e24b4a", lw=2.5))
        el = (surf_cfg["xr"][1] - surf_cfg["xr"][0]) * 0.07
        e1, e2 = evecs[:, 1], evecs[:, 0]
        c_e1 = "#378add" if l1 > 0 else "#ba7517"
        c_e2 = "#378add" if l2 > 0 else "#ba7517"
        ax2.plot([hpx - e1[0] * el, hpx + e1[0] * el], [hpy - e1[1] * el, hpy + e1[1] * el], color=c_e1, lw=2.5, label=f"λ₁={l1:.3f}")
        ax2.plot([hpx - e2[0] * el, hpx + e2[0] * el], [hpy - e2[1] * el, hpy + e2[1] * el], color=c_e2, lw=2.5, ls="--", label=f"λ₂={l2:.3f}")
        ax2.scatter([hpx], [hpy], color="#e24b4a", s=90, zorder=10)
        ax2.set_xlim(surf_cfg["xr"])
        ax2.set_ylim(surf_cfg["yr"])
        ax2.set_title(f"Gradient field + Hessian at ({hpx:.2f}, {hpy:.2f})", fontsize=9, fontweight="bold")
        ax2.legend(fontsize=8, loc="lower left")
        ax2.tick_params(labelsize=7)
        ax2.set_xlabel("x", fontsize=7)
        ax2.set_ylabel("y", fontsize=7)

        x_span = surf_cfg["xr"][1] - surf_cfg["xr"][0]
        y_span = surf_cfg["yr"][1] - surf_cfg["yr"][0]
        patch_half_x = 0.12 * x_span
        patch_half_y = 0.12 * y_span
        px0 = max(surf_cfg["xr"][0], hpx - patch_half_x)
        px1 = min(surf_cfg["xr"][1], hpx + patch_half_x)
        py0 = max(surf_cfg["yr"][0], hpy - patch_half_y)
        py1 = min(surf_cfg["yr"][1], hpy + patch_half_y)
        px = np.linspace(px0, px1, 30)
        py = np.linspace(py0, py1, 30)
        PX, PY = np.meshgrid(px, py)
        PZ = np.clip(f_h(PX, PY), 0, 1)

        ax3d = fig2.add_axes([0.73, 0.50, 0.24, 0.34], projection='3d')
        ax3d.plot_surface(PX, PY, PZ, cmap="Blues_r", linewidth=0, antialiased=True, alpha=0.95)
        probe_z = float(np.clip(f_h(hpx, hpy), 0, 1))
        ax3d.scatter([hpx], [hpy], [probe_z], color="#e24b4a", s=24, depthshade=False)
        ax3d.set_title("Local 3D patch", fontsize=8, pad=2)
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.set_box_aspect((1, 1, 0.55))
        ax3d.view_init(elev=32, azim=-55)
        ax3d.set_facecolor((1, 1, 1, 0.0))

        ax2.indicate_inset_zoom(ax3d, edgecolor="#555555", alpha=0.8)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

# ─────────────────────────────────────────────────────────────
# TAB 3 — Stochastic noise
# ─────────────────────────────────────────────────────────────
with tab3:
    noise_eff = sigma / np.sqrt(batchsz)
    surf_s = SURFACES["Rosenbrock"]
    left, right = st.columns([1, 2.4], gap="small")
    with left:
        st.markdown("**Gradient noise model**")
        st.latex(r"\hat{g} = g(\theta) + \varepsilon,\quad \varepsilon \sim \mathcal{N}\!\left(0,\tfrac{\sigma^2}{B}\right)")
        st.metric("Effective noise σ/√B", f"{noise_eff:.4f}")
        st.caption("Displayed noise is intentionally scaled ×3 for visibility. Treat this panel as illustrative, not as a formal statistical simulator.")
    with right:
        np.random.seed(42)
        paths_clean, _, _, _ = run_sim(surf_s["f"], surf_s["start"], n, lr, b1, b2, 0.01, surf_s["h_fd"], noise=0.0)
        paths_noisy, _, _, _ = run_sim(surf_s["f"], surf_s["start"], n, lr, b1, b2, 0.01, surf_s["h_fd"], noise=noise_eff * 3)
        display = {
            "SGD (full batch)": (paths_clean["SGD"], "#e24b4a"),
            "SGD (mini-batch)": (paths_noisy["SGD"], "#378add"),
            "RMSProp (noisy)": (paths_noisy["RMSProp"], "#1d9e75"),
            "AdamW (noisy)": (paths_noisy["AdamW"], "#ba7517"),
        }
        Xg, Yg, Z = make_grid(surf_s["xr"], surf_s["yr"], 150, "Rosenbrock")
        fig3, ax3 = plt.subplots(figsize=(FW, FH))
        ax3.contourf(Xg, Yg, Z, levels=35, cmap="Blues_r", alpha=0.85)
        ax3.contour(Xg, Yg, Z, levels=15, colors="white", linewidths=0.3, alpha=0.4)
        for name, (path, col) in display.items():
            xs, ys = zip(*path)
            ax3.plot(xs, ys, color=col, lw=1.8, label=name, alpha=0.9)
            ax3.scatter(xs[-1], ys[-1], color=col, s=50, zorder=5)
        ax3.set_xlim(surf_s["xr"])
        ax3.set_ylim(surf_s["yr"])
        ax3.set_title(f"Rosenbrock — effective noise σ/√B = {noise_eff:.3f}", fontsize=9, fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.set_xlabel("x", fontsize=7)
        ax3.set_ylabel("y", fontsize=7)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

# ─────────────────────────────────────────────────────────────
# TAB 4 — Saddle escape
# ─────────────────────────────────────────────────────────────
with tab4:
    saddle_f = lambda x, y: 0.5 * (np.tanh(0.3 * x ** 2 - 0.3 * y ** 2 + 0.05 * x ** 3 + 0.02 * y ** 4) + 1)
    saddle_h = 1e-4
    np.random.seed(7)
    sx = perturb * (np.random.rand() - 0.5)
    sy = perturb * (np.random.rand() - 0.5)
    np.random.seed(42)
    paths_sad, losses_sad, _, _ = run_sim(saddle_f, (sx, sy), n, lr, b1, b2, 0.005, saddle_h, noise=0.001)
    l1s, l2s, *_ = hessian_eigs(saddle_f, 0.0, 0.0, saddle_h * 10)

    left, right = st.columns([1, 2.4], gap="small")
    with left:
        st.markdown("**Saddle at (0, 0)**")
        c1, c2 = st.columns(2)
        c1.metric("λ₁", f"{l1s:.4f}")
        c2.metric("λ₂", f"{l2s:.4f}")
        st.info("Mixed signs → saddle confirmed")
        st.markdown("**Final losses**")
        c1, c2 = st.columns(2)
        for i, name in enumerate(COLORS):
            (c1 if i % 2 == 0 else c2).metric(name, f"{losses_sad[name][-1]:.5f}")

    with right:
        X = np.linspace(-2.5, 2.5, 150)
        Y = np.linspace(-2.5, 2.5, 150)
        Xg, Yg = np.meshgrid(X, Y)
        Z = np.clip(saddle_f(Xg, Yg), 0, 1)
        fig4, axes = plt.subplots(1, 2, figsize=(FW * 1.55, FH))
        ax4 = axes[0]
        ax4.contourf(Xg, Yg, Z, levels=35, cmap="RdYlBu_r", alpha=0.75)
        ax4.contour(Xg, Yg, Z, levels=20, colors="white", linewidths=0.3, alpha=0.4)
        for name, path in paths_sad.items():
            xs, ys = zip(*path)
            ax4.plot(xs, ys, color=COLORS[name], lw=1.8, label=name, alpha=0.9)
            ax4.scatter(xs[-1], ys[-1], color=COLORS[name], s=50, zorder=5)
        ax4.scatter([0], [0], marker="x", color="white", s=100, linewidths=2, zorder=10, label="saddle")
        ax4.set_xlim((-2.5, 2.5))
        ax4.set_ylim((-2.5, 2.5))
        ax4.set_title("Trajectory", fontsize=9, fontweight="bold")
        ax4.legend(fontsize=7)
        ax4.tick_params(labelsize=7)
        ax4.set_xlabel("x", fontsize=7)
        ax4.set_ylabel("y", fontsize=7)

        ax5 = axes[1]
        for name, ll in losses_sad.items():
            ax5.plot(ll, color=COLORS[name], lw=1.8, label=name)
        ax5.axhline(y=0.5, color="gray", ls="--", lw=1, alpha=0.5, label="saddle value")
        ax5.set_title("Loss vs iteration", fontsize=9, fontweight="bold")
        ax5.legend(fontsize=7)
        ax5.tick_params(labelsize=7)
        ax5.set_xlabel("Iteration", fontsize=7)
        ax5.set_ylabel("f(x,y)", fontsize=7)
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

# ─────────────────────────────────────────────────────────────
# TAB 5 — Convergence
# ─────────────────────────────────────────────────────────────
with tab5:
    left, right = st.columns([1, 2.4], gap="small")
    with left:
        st.markdown("**Final loss**")
        c1, c2 = st.columns(2)
        for i, name in enumerate(COLORS):
            (c1 if i % 2 == 0 else c2).metric(name, f"{losses[name][-1]:.6f}")

        st.markdown("**Experiment summary**")
        summary_rows = []
        for name in COLORS:
            path = np.array(paths[name])
            step_deltas = np.diff(path, axis=0)
            path_length = float(np.sum(np.linalg.norm(step_deltas, axis=1)))
            avg_step = float(np.mean(internals[name]["step_norm"]))
            summary_rows.append({
                "Optimizer": name,
                "Path length": round(path_length, 4),
                "Avg |step|": round(avg_step, 4),
                "Final loss": round(float(losses[name][-1]), 6),
            })
        st.dataframe(summary_rows, use_container_width=True, hide_index=True)
        st.caption("Use presets to compare the same scenario reproducibly across optimizers.")
    with right:
        fig5, ax6 = plt.subplots(figsize=(FW, FH))
        ls_ = {"SGD": "-", "Momentum": "--", "RMSProp": "-.", "AdamW": ":"}
        for name, ll in losses.items():
            ax6.plot(ll, color=COLORS[name], lw=2.0, ls=ls_[name], label=name)
        if log_y:
            ax6.set_yscale("log")
        ax6.set_title(f"Convergence — {surf_name}", fontsize=9, fontweight="bold")
        ax6.set_xlabel("Iteration", fontsize=8)
        ax6.set_ylabel("f(x,y)", fontsize=8)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.2)
        ax6.tick_params(labelsize=7)
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

# ─────────────────────────────────────────────────────────────
# TAB 6 — Optimizer Internals
# ─────────────────────────────────────────────────────────────
with tab6:
    st.markdown("### Why the paths differ")
    st.caption("This panel exposes the hidden optimizer state. It turns the app from a trajectory viewer into an explanation tool.")

    step_idx = selected_step - 1
    rows = []
    for name in COLORS:
        rows.append({
            "Optimizer": name,
            "Loss @ step": losses[name][step_idx],
            "|grad|": internals[name]["grad_norm"][step_idx],
            "|step|": internals[name]["step_norm"][step_idx],
            "effective LR": internals[name]["eff_lr"][step_idx],
            "|velocity|": internals[name]["velocity_norm"][step_idx],
            "|1st moment|": internals[name]["moment_norm"][step_idx],
            "|2nd moment|": internals[name]["second_moment_norm"][step_idx],
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1.1, 1.9], gap="small")
    with c1:
        st.markdown("**Interpretation guide**")
        st.markdown(
            "- **|grad|**: local steepness of the surface.\n"
            "- **|step|**: actual move the optimizer makes.\n"
            "- **effective LR**: move size divided by gradient size.\n"
            "- **|velocity|**: accumulated momentum state.\n"
            "- **|2nd moment|**: adaptive denominator state used by RMSProp/AdamW."
        )
        st.info(
            f"At iteration {selected_step}, compare **|grad|** against **|step|**. "
            "Large gradients with small steps usually mean damping or adaptation; large steps with modest gradients usually mean momentum carryover or low adaptive denominator."
        )

    with c2:
        fig_int, axes = plt.subplots(2, 2, figsize=(FW * 1.45, FH * 1.35))
        metrics = [
            ("grad_norm", "Gradient norm"),
            ("step_norm", "Step norm"),
            ("velocity_norm", "Velocity norm"),
            ("second_moment_norm", "Second-moment norm"),
        ]
        for ax, (metric_key, title) in zip(axes.flatten(), metrics):
            for name, color in COLORS.items():
                ax.plot(internals[name][metric_key], color=color, lw=1.9, label=name)
            ax.axvline(selected_step - 1, color="gray", ls="--", lw=1.0, alpha=0.6)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.2)
        axes[0, 0].legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig_int, use_container_width=True)
        plt.close(fig_int)

st.markdown("---")
st.markdown("## 📚 Theory Reference")
st.caption("Click any topic to expand. These notes are static and safe for presentation.")

with st.expander("1 · SGD and gradient descent fundamentals"):
    st.markdown(r"""
**Vanilla SGD** performs the parameter update:
$$\theta_{t+1} = \theta_t - \alpha \nabla f(\theta_t)$$

- **Convex** convergence rate: $O(1/\sqrt{T})$ with constant LR; $O(1/T)$ with decaying LR ~ 1/t.
- On **non-convex** surfaces, convergence is generally to a stationary point under assumptions, not necessarily to the global minimum.
- Highly sensitive to local curvature and ill-conditioning.
- **LR too large** → instability; **LR too small** → slow progress.
    """)

with st.expander("2 · Momentum: Polyak and Nesterov"):
    st.markdown(r"""
**Heavy-ball momentum** in this demo uses:
$$v_t = \beta_1 v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \alpha v_t$$

- Momentum carries directional information across steps.
- It often helps in long valleys by smoothing zig-zag motion.
- Different libraries use slightly different conventions; compare hyperparameters carefully.
    """)

with st.expander("3 · RMSProp: adaptive learning rates"):
    st.markdown(r"""
**RMSProp** keeps an exponential moving average of squared gradients:
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \qquad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t}+\epsilon}g_t$$

- Coordinates with persistently large gradients get smaller effective steps.
- Coordinates with smaller gradients get relatively larger steps.
- This is why RMSProp often behaves better than SGD on badly scaled surfaces.
    """)

with st.expander("4 · Adam and AdamW: bias correction and decoupled weight decay"):
    st.markdown(r"""
**Adam** combines first- and second-moment accumulation with bias correction.

**AdamW** applies weight decay separately from the adaptive gradient term:
$$\theta \leftarrow (1-\alpha\lambda)\theta - \alpha\frac{\hat m}{\sqrt{\hat v}+\epsilon}$$

- The decay acts on the parameter directly.
- It does **not** get rescaled by the adaptive denominator.
- This decoupling is why AdamW is usually preferred over naive Adam + L2 regularization.
    """)
