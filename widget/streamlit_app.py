"""
Live IRRBB hedging widget (Streamlit port of the Bokeh app) — slide the problem from
"LQ wins" to "RL wins".

Reuses the exact same vectorised recompute core (compute.py); only the UI/plotting is
Streamlit + Plotly instead of a Bokeh server, so it deploys to Streamlit Cloud with no
long-running websocket process. The recompute is ~0.15 s, so Streamlit's rerun-on-change
model is plenty responsive.

Run locally:   streamlit run widget/streamlit_app.py
Deploy:        push to GitHub, point Streamlit Cloud at widget/streamlit_app.py
               (the repo must include _sweep/nb_cache.pkl and _sweep/impulse_results.json)
"""
import os, sys
import numpy as np
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
import compute

C0, C_OPT, C_RL, C_GREY = "#1f77b4", "#d62728", "#2ca02c", "#888888"

st.set_page_config(page_title="IRRBB: RL vs LQ", layout="wide")


@st.cache_resource
def get_model():
    return compute.Model()


M = get_model()

# ----------------------------------------------------------------- sidebar (knobs)
st.sidebar.markdown("**Cost & nonlinearity knobs**")
log_kf = st.sidebar.slider("log₁₀ fixed cost per trade (kappa_fixed)", -4.0, -1.5,
                           float(np.log10(4e-3)), 0.05)
band = st.sidebar.slider("no-trade band half-width (manual)", 0.0, 40.0, 18.0, 0.5)
st.sidebar.markdown("**Balance-sheet structure**")
beta_d = st.sidebar.slider("deposit beta (pass-through)", 0.2, 1.0, 0.8, 0.05)
eta_L = st.sidebar.slider("deposit flight  eta_L", 0.0, 8.0, 1.0, 0.25)
rho1 = st.sidebar.slider("prepay convexity  rho1", 0.0, 1.0, 0.7, 0.05)
prepay_offset = st.sidebar.slider("prepay trigger depth  prepay_offset", 0.0, 0.03, 0.015, 0.001)
d_floor = st.sidebar.slider("deposit-rate floor  d_floor", 0.0, 0.03, 0.018, 0.001)
w_down = st.sidebar.slider("downside / CVaR weight  w_down", 1.0, 10.0, 1.0, 0.5)

kf = 10 ** log_kf
r = M.recompute(dict(kappa_fixed=kf, beta_d=beta_d, eta_L=eta_L, rho1=rho1,
                     prepay_offset=prepay_offset, d_floor=d_floor, w_down=w_down))
bands = r["bands"]

# ----------------------------------------------------------------- header
st.markdown("## IRRBB hedging: optimal control vs reinforcement learning")
st.markdown(
    "RL beats LQ **iff the optimal policy is nonlinear**. Here a **fixed cost per trade** creates a "
    "nonlinear no-trade (s,S) band that LQ's smooth quadratic control cannot express. Raise "
    "`kappa_fixed` (sidebar) and watch RL overtake LQ.")

# ----------------------------------------------------------------- fig 1: cost vs band
mb_total = float(np.interp(band, bands, r["total"]))
f1 = go.Figure()
f1.add_trace(go.Scatter(x=bands, y=r["total"], mode="lines", line=dict(color=C0, width=2),
                        name="total cost"))
f1.add_trace(go.Scatter(x=[0.0, r["opt"]["band"], band],
                        y=[r["cont"]["total"], r["opt"]["total"], mb_total],
                        mode="markers", marker=dict(size=12, color=[C0, C_OPT, C_GREY]),
                        text=["LQ continuous (band 0)", f"optimal band ({r['opt']['band']:.0f})",
                              "your band"], name="", showlegend=False))
f1.add_vline(x=band, line=dict(color=C_GREY, dash="dash"))
f1.update_layout(title="Total economic cost vs no-trade band", height=320,
                 xaxis_title="no-trade band half-width", yaxis_title="total cost",
                 margin=dict(l=10, r=10, t=40, b=10))

# ----------------------------------------------------------------- fig 2: cost vs kappa frontier
f2 = go.Figure()
f2.add_trace(go.Scatter(x=r["kappa_grid"], y=r["opt_frontier"], mode="lines",
                        line=dict(color=C_RL, width=2), name="impulse optimum = what RL learns"))
f2.add_trace(go.Scatter(x=r["kappa_grid"], y=r["cont_frontier"], mode="lines",
                        line=dict(color=C0, width=2), fill="tonexty",
                        fillcolor="rgba(44,160,44,0.15)", name="LQ (continuous rebalancing)"))
f2.add_vline(x=kf, line=dict(color=C_GREY, dash="dash"))
f2.update_layout(title="Cost vs fixed trading cost: where does RL overtake LQ?", height=320,
                 xaxis_title="fixed cost per trade (kappa_fixed, log)", yaxis_title="total cost",
                 xaxis_type="log", legend=dict(x=0.02, y=0.98, font=dict(size=10)),
                 margin=dict(l=10, r=10, t=40, b=10))

c1, c2 = st.columns(2)
c1.plotly_chart(f1, use_container_width=True)
c2.plotly_chart(f2, use_container_width=True)

# ----------------------------------------------------------------- fig 3: hedge trajectory
W = slice(0, 160)
t = np.arange(M.T)[W]
hcont = M.band_traj(r["hstar"], 0.0)[W]
hband = M.band_traj(r["hstar"], r["opt"]["band"])[W]
f3 = go.Figure()
f3.add_trace(go.Scatter(x=t, y=r["hstar"][0][W], mode="lines", line=dict(color=C_GREY, width=1),
                        name="variance-min target h*"))
f3.add_trace(go.Scatter(x=t, y=hcont, mode="lines", line=dict(color=C0, width=1), opacity=0.6,
                        name="LQ continuous"))
f3.add_trace(go.Scatter(x=t, y=hband, mode="lines", line=dict(color=C_RL, width=2),
                        name="RL (s,S) band"))
f3.update_layout(title="The policy: continuous LQ trades every month; RL holds inside an (s,S) band",
                 height=280, xaxis_title="month", yaxis_title="hedge notional h",
                 legend=dict(x=0.78, y=0.98, font=dict(size=10)), margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(f3, use_container_width=True)

# ----------------------------------------------------------------- readout
verdict = ("LQ is essentially optimal here — reach for the Riccati solution."
           if r["gap_pct"] < 8 else
           "RL earns its keep — the optimum is a nonlinear no-trade band.")
m1, m2, m3 = st.columns(3)
m1.metric("optimal no-trade band", f"{r['opt']['band']:.0f}")
m2.metric("months traded (vs LQ's 100%)", f"{r['opt']['freq']:.0%}")
m3.metric("impulse optimum vs continuous LQ", f"{r['gap_pct']:.0f}% cheaper")
st.markdown(f"At **kappa_fixed = {kf:.1e}** — *{verdict}*")

if M.rl is not None:
    seeds = M.rl["rl_seeds"]
    tot = np.array([seeds[s]["total"] for s in seeds])
    win = int((tot < M.rl["continuous_LQ"]["total"]).sum())
    med = 100 * (1 - np.median(tot) / M.rl["continuous_LQ"]["total"])
    st.caption(f"Validation: a SAC agent trained at kappa_fixed=4e-3 (default knobs) hit this optimum "
               f"in {win}/{len(tot)} seeds (median {med:.0f}% below continuous LQ).")
