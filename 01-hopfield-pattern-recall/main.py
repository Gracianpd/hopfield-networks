"""
Red de Hopfield (p=1) — evolución temporal del overlap según nivel de ruido
Verifica la regla 2n < N para la recuperación del patrón (letra "A").
Versión aleatoria: cada ejecución produce resultados distintos.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Red y patrón ─────────────────────────────────────────────────────────────

N_ROWS, N_COLS = 10, 10
N = N_ROWS * N_COLS

A_GRID = [
    "....XX....",
    "....XX....",
    "...X..X...",
    "...X..X...",
    "..XXXXXX..",
    "..X....X..",
    ".X......X.",
    ".X......X.",
    "X........X",
    "X........X",
]
xi = np.array(
    [[1 if c == "X" else -1 for c in row] for row in A_GRID],
    dtype=float
).flatten()

# Aprendizaje: regla de Hebb  J_ij = (1/N) xi_i xi_j,  J_ii = 0
J = np.outer(xi, xi) / N
np.fill_diagonal(J, 0.0)

# ── Funciones ────────────────────────────────────────────────────────────────

def corrupt(n_flip, rng):
    s = xi.copy()
    s[rng.choice(N, size=n_flip, replace=False)] *= -1
    return s

def overlap(S):
    return float(np.dot(S, xi) / N)

def run_async(S_init, rng):
    """Dinámica asíncrona hasta convergencia. Devuelve overlap en cada paso."""
    S = S_init.copy()
    ov = [overlap(S)]
    for _ in range(20):
        S_prev = S.copy()
        for i in rng.permutation(N):
            S[i] = 1.0 if J[i] @ S > 0 else -1.0
            ov.append(overlap(S))
        if np.array_equal(S, S_prev):
            break
    return np.array(ov)

# ── Simulación ───────────────────────────────────────────────────────────────

NOISE_LEVELS = [5, 10, 20, 30, 40, 45, 49, 50, 51, 55, 60]

# Semilla aleatoria distinta en cada ejecución
rng = np.random.default_rng()

results = {}
for n in NOISE_LEVELS:
    S0 = corrupt(n, rng)
    ov = run_async(S0, rng)
    results[n] = ov

# ── Figura ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))

for n, ov in results.items():
    pct   = 100 * n / N
    ok    = ov[-1] > 0.99
    color = plt.cm.Blues(0.4 + 0.6 * n / 50) if ok else plt.cm.Reds(0.4 + 0.6 * (n - 50) / 15)
    ls    = "-" if ok else "--"
    label = f"n={n} ({pct:.0f}%)  {'✓' if ok else '✗'}"
    ax.plot(ov, color=color, lw=2, ls=ls, label=label)

ax.axhline( 1.0, color="green", ls=":", lw=1.2, alpha=0.6, label="m = +1  (ξ recuperado)")
ax.axhline(-1.0, color="red",   ls=":", lw=1.2, alpha=0.4, label="m = −1  (−ξ, fallo)")
ax.axhline( 0.0, color="gray",  ls=":", lw=0.8, alpha=0.5)

ax.set_xlabel("Paso", fontsize=12)
ax.set_ylabel(r"Overlap  $m(t)$", fontsize=12)
ax.set_title("Evolución dinámica del overlap — Red de Hopfield (p=1, N=100)", fontsize=12)
ax.legend(loc="center right", fontsize=9, framealpha=0.9)
ax.set_ylim(-1.1, 1.15)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()