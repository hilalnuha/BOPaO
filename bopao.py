# ============================================================
# BOPaO vs PSO vs GA - Standard Function Benchmark
# ============================================================
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import gamma as Gamma, pi, sin

# ---------- Benchmark functions (vectorized: X is (N, D) -> (N,)) ----------
def sphere(X):        return np.sum(X**2, axis=1)
def schwefel222(X):   return np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)
def rosenbrock(X):    return np.sum(100*(X[:, 1:] - X[:, :-1]**2)**2 + (X[:, :-1] - 1)**2, axis=1)
def rastrigin(X):     return 10*X.shape[1] + np.sum(X**2 - 10*np.cos(2*pi*X), axis=1)
def ackley(X):
    return (-20*np.exp(-0.2*np.sqrt(np.mean(X**2, axis=1)))
            - np.exp(np.mean(np.cos(2*pi*X), axis=1)) + 20 + np.e)
def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return 1 + np.sum(X**2, axis=1)/4000 - np.prod(np.cos(X/np.sqrt(i)), axis=1)

# name: (function, lower bound, upper bound)
BENCH = {
    "Sphere (F1)":        (sphere,       -100.0,  100.0),
    "Schwefel 2.22 (F2)": (schwefel222,   -10.0,   10.0),
    "Rosenbrock (F3)":    (rosenbrock,    -30.0,   30.0),
    "Rastrigin (F4)":     (rastrigin,      -5.12,   5.12),
    "Ackley (F5)":        (ackley,        -32.0,   32.0),
    "Griewank (F6)":      (griewank,     -600.0,  600.0),
}

# ---------- Levy flight (Mantegna's algorithm) ----------
def levy(shape, lam, rng):
    sigma = (Gamma(1 + lam) * sin(pi*lam/2) /
             (Gamma((1 + lam)/2) * lam * 2**((lam - 1)/2))) ** (1/lam)
    u = rng.normal(0, sigma, shape)
    v = rng.normal(0, 1, shape)
    return u / np.abs(v)**(1/lam)

# ============================================================
# 1) BOPaO
# ============================================================
def bopao(f, lb, ub, dim, pop=30, iters=500, seed=0,
          p_exp=0.3, p_dis=0.5, p_imi=0.2,
          alpha=0.1, lam=1.5, beta=1.0, gamma=0.01, delta=0.5):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, (pop, dim))
    fit = f(X)
    b = np.argmin(fit)
    xbest, fbest = X[b].copy(), fit[b]
    curve = np.empty(iters)

    for t in range(iters):
        phase = rng.choice(3, size=pop, p=[p_exp, p_dis, p_imi])[:, None]
        diff  = xbest - X

        X_exp = X + alpha * levy((pop, dim), lam, rng) * diff            # Phase 1
        theta = rng.random((pop, 1))
        X_dis = X + beta * np.cos(2*pi*theta) * np.exp(-gamma*t) * diff  # Phase 2
        x_rand = X[rng.integers(0, pop, pop)]
        X_imi = xbest + delta * rng.random((pop, dim)) * (xbest - x_rand)  # Phase 3

        Xn = np.where(phase == 0, X_exp, np.where(phase == 1, X_dis, X_imi))
        Xn = np.clip(Xn, lb, ub)
        fn = f(Xn)

        better = fn < fit                                                # greedy selection
        X[better], fit[better] = Xn[better], fn[better]

        b = np.argmin(fit)
        if fit[b] < fbest:
            xbest, fbest = X[b].copy(), fit[b]
        curve[t] = fbest
    return xbest, fbest, curve

# ============================================================
# 2) PSO (inertia-weight variant, Kennedy & Eberhart)
# ============================================================
def pso(f, lb, ub, dim, pop=30, iters=500, seed=0,
        w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, (pop, dim))
    vmax = 0.2 * (ub - lb)
    V = rng.uniform(-vmax, vmax, (pop, dim))
    fit = f(X)
    pbest, pfit = X.copy(), fit.copy()
    b = np.argmin(pfit)
    gbest, gfit = pbest[b].copy(), pfit[b]
    curve = np.empty(iters)

    for t in range(iters):
        w = w_max - (w_max - w_min) * t / max(iters - 1, 1)             # linearly decreasing inertia
        r1, r2 = rng.random((pop, dim)), rng.random((pop, dim))
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)                 # velocity update
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, lb, ub)                                      # position update
        fit = f(X)

        better = fit < pfit
        pbest[better], pfit[better] = X[better], fit[better]
        b = np.argmin(pfit)
        if pfit[b] < gfit:
            gbest, gfit = pbest[b].copy(), pfit[b]
        curve[t] = gfit
    return gbest, gfit, curve

# ============================================================
# 3) GA (real-coded: roulette selection, one-point crossover,
#        Gaussian mutation, elitism)
# ============================================================
def ga(f, lb, ub, dim, pop=30, iters=500, seed=0,
       pc=0.8, pm=0.1, sigma_frac=0.1, n_elite=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lb, ub, (pop, dim))
    fit = f(X)
    b = np.argmin(fit)
    xbest, fbest = X[b].copy(), fit[b]
    sigma = sigma_frac * (ub - lb)
    cols = np.arange(dim)[None, :]
    curve = np.empty(iters)

    for t in range(iters):
        # Roulette wheel for minimization: weight = (f_max - f) so lower cost -> higher chance
        wts = (fit.max() - fit) + 1e-12
        prob = wts / wts.sum()
        p1 = X[rng.choice(pop, pop, p=prob)]
        p2 = X[rng.choice(pop, pop, p=prob)]

        # One-point crossover: child = [p1[:k] | p2[k:]]
        k = rng.integers(1, max(dim, 2), (pop, 1))
        child = np.where(cols < k, p1, p2)
        no_cx = rng.random(pop) >= pc
        child[no_cx] = p1[no_cx]

        # Gaussian mutation: g' = g + N(0, sigma^2) with probability pm per gene
        mut = rng.random((pop, dim)) < pm
        child = np.clip(child + mut * rng.normal(0, sigma, (pop, dim)), lb, ub)
        cfit = f(child)

        # Elitism: carry the best individuals of the old generation over
        elite = np.argsort(fit)[:n_elite]
        worst = np.argsort(cfit)[-n_elite:]
        child[worst], cfit[worst] = X[elite], fit[elite]

        X, fit = child, cfit
        b = np.argmin(fit)
        if fit[b] < fbest:
            xbest, fbest = X[b].copy(), fit[b]
        curve[t] = fbest
    return xbest, fbest, curve

# ============================================================
# Experiment (same pop, iterations, and seeds for every method)
# ============================================================
METHODS = {"BOPaO": bopao, "PSO": pso, "GA": ga}
DIM, POP, ITERS, RUNS = 30, 30, 500, 10

rows, curves = [], {}
for fname, (f, lb, ub) in BENCH.items():
    for mname, algo in METHODS.items():
        vals, times, cs = [], [], []
        for r in range(RUNS):
            t0 = time.perf_counter()
            _, fb, c = algo(f, lb, ub, DIM, POP, ITERS, seed=r)
            times.append(time.perf_counter() - t0)
            vals.append(fb); cs.append(c)
        vals = np.array(vals)
        curves[(fname, mname)] = np.mean(cs, axis=0)
        rows.append({"Function": fname, "Method": mname,
                     "Best": vals.min(), "Worst": vals.max(),
                     "Mean": vals.mean(), "Std": vals.std(),
                     "Mean time (s)": np.mean(times)})

df = pd.DataFrame(rows).set_index(["Function", "Method"])
pd.set_option("display.float_format", lambda v: f"{v:.4e}")
pd.set_option("display.width", 140)
print(f"D={DIM}, pop={POP}, iters={ITERS}, runs={RUNS}\n")
print(df)

# ---------- Mean rank per function (1 = best mean) ----------
rank_tbl = (df["Mean"].unstack("Method")
              .rank(axis=1, method="min")
              .loc[list(BENCH.keys())])
rank_tbl.loc["Average rank"] = rank_tbl.mean()
print("\nRank by mean fitness (1 = best):\n")
print(rank_tbl.to_string(float_format=lambda v: f"{v:.2f}"))

# ---------- Convergence plots (mean of runs) ----------
colors = {"BOPaO": "crimson", "PSO": "darkorange", "GA": "seagreen"}
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
for ax, fname in zip(axes.ravel(), BENCH.keys()):
    for mname in METHODS:
        ax.semilogy(np.maximum(curves[(fname, mname)], 1e-300),
                    color=colors[mname], lw=1.8, label=mname)
    ax.set_title(fname); ax.set_xlabel("Iteration"); ax.set_ylabel("Best fitness")
    ax.grid(True, which="both", alpha=0.3)
axes[0, 0].legend()
plt.suptitle(f"Convergence comparison on standard benchmarks (mean of {RUNS} runs)")
plt.tight_layout()
plt.show()
