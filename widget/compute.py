"""
Fast, fully-vectorised recompute core for the live IRRBB RL-vs-LQ widget.

The whole RL-vs-LQ story reduces to one fast computation: given the (rate-dependent)
balance sheet and a fixed trading cost, what is the optimal no-trade band, and how far
does the impulse-control optimum (what RL learns) pull ahead of continuous LQ rebalancing?

Everything here is closed-form / vectorised so a slider change recomputes in < 0.1 s:
  * AFNS short/long yields along precomputed factor paths           (vectorised)
  * deposit-balance / asset-book-yield dynamics                     (vectorised over paths)
  * the variance-minimising LINEAR hedge target h*(state)           (closed-form OLS)
  * a sweep over no-trade band widths -> (variance, trade-freq)     (vectorised (s,S) tracking)
The LQ baseline is band=0 (rebalance every step); the RL-achievable optimum is the best band.
"""
import os, json
import numpy as np

_HERE = os.path.dirname(__file__)
_CACHE = os.path.join(_HERE, "..", "_sweep", "nb_cache.pkl")
_IMP = os.path.join(_HERE, "..", "_sweep", "impulse_results.json")


# ---- AFNS yield adjustment (matches the notebook / nb_cache exactly) ----
def _afns_adj(tau, lam, s1, s2, s3):
    tau = max(float(tau), 1e-8); x = lam * tau
    e1, e2 = np.exp(-x), np.exp(-2 * x)
    I1 = (s1 ** 2 / 6.0) * tau ** 2
    I2 = s2 ** 2 * (1 / (2 * lam ** 2) - (1 - e1) / (lam ** 3 * tau) + (1 - e2) / (4 * lam ** 3 * tau))
    I3 = s3 ** 2 * (1 / (2 * lam ** 2) + e1 / lam ** 2 - tau * e2 / (4 * lam)
                    - 3 * e2 / (4 * lam ** 2) - 2 * (1 - e1) / (lam ** 3 * tau)
                    + 5 * (1 - e2) / (8 * lam ** 3 * tau))
    return I1 + I2 + I3


DEFAULTS = dict(  # structural + cost knobs exposed as sliders
    A0=80.0, L0=150.0, beta_d=0.8, d_floor=0.018, eta_L=1.0, rho_L=0.15,
    rho0=0.05, rho1=0.70, prepay_slope=1200.0, prepay_offset=0.015,
    kappa_fixed=4e-3, kappa_prop=1e-4, w_down=1.0,
)
DT, TAU_L, TAU_A = 1.0 / 12.0, 1.0, 3.0
BANDS = np.concatenate([[0.0], np.linspace(0.5, 40, 28)])


class Model:
    """Loads the cached AFNS setup once; recomputes the band/cost story per param set."""

    def __init__(self, n_paths=120, T=200, seed=12345):
        import pickle
        with open(_CACHE, "rb") as f:
            d = pickle.load(f)
        self.lam, self.s1, self.s2, self.s3 = d["lam"], d["sig1"], d["sig2"], d["sig3"]
        self.X_ref, self.c_hat, self.Phi, self.Sig = d["X_ref"], d["c_hat"], d["Phi_hat"], d["Sigma_hat"]
        # NS loadings at the two tenors + the AFNS adjustments (constants)
        self._ld = {tau: self._loadings(tau) for tau in (TAU_L, TAU_A)}
        self._adj = {tau: _afns_adj(tau, self.lam, self.s1, self.s2, self.s3) for tau in (TAU_L, TAU_A)}
        self.y_ref_s = self._y_ref(TAU_L); self.y_ref_l = self._y_ref(TAU_A)
        self.K = self.y_ref_s
        self._sim_paths(n_paths, T, seed)
        try:
            self.rl = json.load(open(_IMP))
        except Exception:
            self.rl = None

    def _loadings(self, tau):
        lam = self.lam
        b2 = (1 - np.exp(-lam * tau)) / (lam * tau)
        return np.array([1.0, b2, b2 - np.exp(-lam * tau)])

    def _y_ref(self, tau):
        return float(self.X_ref @ self._ld[tau] - self._adj[tau])

    def _sim_paths(self, n, T, seed):
        rng = np.random.default_rng(seed)
        X = np.zeros((n, T, 3)); X[:, 0] = self.X_ref
        L = np.linalg.cholesky(self.Sig)
        for t in range(T - 1):
            eps = rng.standard_normal((n, 3)) @ L.T
            X[:, t + 1] = self.c_hat + X[:, t] @ self.Phi.T + eps
        self.X = X
        # vectorised yields along all paths
        self.ys = X @ self._ld[TAU_L] - self._adj[TAU_L]   # (n,T)
        self.yl = X @ self._ld[TAU_A] - self._adj[TAU_A]
        self.n, self.T = n, T

    # ---- the live recompute ----
    def recompute(self, p):
        p = {**DEFAULTS, **p}
        ys, yl = self.ys, self.yl
        n, T = self.n, self.T
        # balances (vectorised over paths, sequential in time)
        L = np.empty((n, T)); yab = np.empty((n, T))
        L[:, 0] = p["L0"]; yab[:, 0] = self.y_ref_l
        for t in range(T - 1):
            L_tgt = p["L0"] * np.exp(-p["eta_L"] * (ys[:, t] - self.y_ref_s))
            L[:, t + 1] = L[:, t] + p["rho_L"] * (L_tgt - L[:, t])
            rho_yA = p["rho0"] + p["rho1"] / (1 + np.exp(-p["prepay_slope"] *
                       (self.y_ref_l - p["prepay_offset"] - yl[:, t])))
            yab[:, t + 1] = yab[:, t] + rho_yA * (yl[:, t] - yab[:, t])

        d_t = np.maximum(p["d_floor"], p["beta_d"] * ys)
        nii_unh = p["A0"] * yab * DT - L * d_t * DT
        payoff = (self.K - ys) * DT
        target = (p["A0"] * self.y_ref_l - p["L0"] * max(p["d_floor"], p["beta_d"] * self.y_ref_s)) * DT

        # closed-form variance-min LINEAR hedge target h*(state) = c . z
        z = np.stack([np.ones_like(ys), L - p["L0"], ys - self.y_ref_s, yab - self.y_ref_l], -1)  # (n,T,4)
        g = (z * payoff[..., None]).reshape(-1, 4)                  # features
        y = nii_unh.reshape(-1)
        gc = g - g.mean(0); yc = y - y.mean()
        c = np.linalg.lstsq(gc, -yc, rcond=None)[0]                 # min Var(nii_unh + g.c)
        hstar = (z @ c)                                            # (n,T) target notional

        # band sweep: (s,S) tracking, vectorised over paths
        w = p["w_down"]
        bands = BANDS
        var = np.empty(bands.size); freq = np.empty(bands.size); turn = np.empty(bands.size)
        for bi, band in enumerate(bands):
            h = np.zeros(n); trades = 0.0; tot = 0.0; absu = 0.0
            for t in range(T - 1):
                gap = hstar[:, t] - h
                m = np.abs(gap) > band
                u = np.where(m, gap, 0.0)
                h = h + u
                trades += m.sum(); absu += np.abs(u).sum()
                nii = nii_unh[:, t] + h * payoff[:, t]
                dev = nii - target
                ww = np.where(dev < 0, w, 1.0)
                tot += np.sum(ww * dev * dev)
            cnt = n * (T - 1)
            var[bi] = tot / cnt; freq[bi] = trades / cnt; turn[bi] = absu / cnt

        kf, kp = p["kappa_fixed"], p["kappa_prop"]
        total = var + kf * freq + kp * turn
        i_opt = int(np.argmin(total))
        cont = dict(total=float(total[0]), var=float(var[0]), freq=float(freq[0]))
        opt = dict(band=float(bands[i_opt]), total=float(total[i_opt]),
                   var=float(var[i_opt]), freq=float(freq[i_opt]))
        gap_pct = 100 * (1 - opt["total"] / cont["total"])

        # cost-vs-kappa frontier (continuous LQ vs impulse optimum), using cached var/freq
        kg = np.geomspace(2e-4, 3e-2, 40)
        cont_f = var[0] + kg * freq[0] + kp * turn[0]
        opt_f = np.array([np.min(var + k * freq + kp * turn) for k in kg])

        return dict(bands=bands, total=total, var=var, freq=freq,
                    cont=cont, opt=opt, gap_pct=gap_pct, target=float(target),
                    kappa_grid=kg, cont_frontier=cont_f, opt_frontier=opt_f,
                    hstar=hstar, nii_unh=nii_unh, payoff=payoff, kappa_fixed=kf)


    @staticmethod
    def band_traj(hstar, band, idx=0):
        """hedge inventory path under an (s,S) band of half-width `band` for one path."""
        hs = hstar[idx]; h = 0.0; out = np.empty(hs.size)
        for t in range(hs.size):
            if abs(hs[t] - h) > band:
                h = hs[t]
            out[t] = h
        return out


if __name__ == "__main__":
    m = Model()
    r = m.recompute({})
    print(f"default params: continuous LQ total={r['cont']['total']:.4e} (freq {r['cont']['freq']:.2f})")
    print(f"optimal band={r['opt']['band']:.1f} total={r['opt']['total']:.4e} (freq {r['opt']['freq']:.2f})")
    print(f"impulse optimum beats continuous LQ by {r['gap_pct']:.1f}%  "
          f"(notebook gate at kappa=4e-3 was ~53%)")
    for kf in [1e-3, 4e-3, 1e-2]:
        rr = m.recompute({"kappa_fixed": kf})
        print(f"  kappa_fixed={kf:.0e}: opt band={rr['opt']['band']:.1f}, gap {rr['gap_pct']:.1f}%")
