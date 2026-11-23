# -*- coding: utf-8 -*-
"""
Problema 4 — Redes atractoras diluidas (continuación del Problema 3).
CONVENCIÓN CANNING–GARDNER: dilución de GRADO FIJO.

Todas las neuronas tienen exactamente el mismo número de conexiones:

    sum_j D_ij = K      para todo i        ("same connectivity at each site")
    w = K / N                              ratio de conectividad
    1 - w                                  fracción de sinapsis destruidas

La máscara D_ij es simétrica (D_ij = D_ji) y sin autoacoplo (D_ii = 0), es
decir, un grafo K-regular aleatorio. La dinámica es la del Problema 3 pero
restringida a las conexiones supervivientes:

    S_i(t+1) = sign( sum_j D_ij J_ij S_j )
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()          # PCG64
N_LADO = 10
N = N_LADO * N_LADO                     # 100 neuronas

# ---------------------------------------------------------------------------
# 1. Patrón "A" (10x10) y regla de Hebb  —  idéntico al Problema 3
# ---------------------------------------------------------------------------
A_BITMAP = [
    "0001111000",
    "0011001100",
    "0110000110",
    "0110000110",
    "0111111110",
    "0110000110",
    "0110000110",
    "0110000110",
    "0110000110",
    "0000000000",
]

def bitmap_a_patron(bitmap):
    grid = np.array([[1 if c == "1" else -1 for c in fila] for fila in bitmap],
                    dtype=int)
    return grid.reshape(-1)

xi = bitmap_a_patron(A_BITMAP)

def matriz_hebb(patrones):
    """J_ij = (1/N) sum_mu xi_i^mu xi_j^mu,  J_ii = 0."""
    P = np.atleast_2d(patrones).astype(float)
    J = (P.T @ P) / N
    np.fill_diagonal(J, 0.0)
    return J

J = matriz_hebb(xi)

# ---------------------------------------------------------------------------
# 2. DILUCIÓN DE GRADO FIJO: grafo K-regular aleatorio simétrico
#    (hipótesis de Canning–Gardner: misma conectividad en todos los sitios)
# ---------------------------------------------------------------------------
import networkx as nx

def mascara_regular(N, K, rng):
    """Devuelve D_ij binaria, simétrica, D_ii=0 y con sum_j D_ij = K EXACTO,
       generada como un grafo K-regular aleatorio con NetworkX.
       Para K denso se genera el grafo complementario (mucho más rápido)."""
    if K <= 0:
        return np.zeros((N, N))
    if K >= N - 1:                                  # red completa
        return np.ones((N, N)) - np.eye(N)
    if (N * K) % 2:
        raise ValueError("N*K debe ser par para que exista un grafo K-regular")

    semilla = int(rng.integers(2**31))
    if K > (N - 1) // 2:                             # denso -> complementario
        G = nx.random_regular_graph(N - 1 - K, N, seed=semilla)
        return 1.0 - np.eye(N) - nx.to_numpy_array(G)
    G = nx.random_regular_graph(K, N, seed=semilla)
    return nx.to_numpy_array(G)


def K_de_w(w):
    """Conexiones por neurona correspondientes a un ratio de conectividad w."""
    return int(min(round(w * N), N - 1))

def W_diluida(K, rng):
    """Matriz efectiva D*J para K conexiones por neurona."""
    if K >= N - 1:
        return J
    return mascara_regular(N, K, rng) * J

# ---------------------------------------------------------------------------
# 3. Observables y corrupción  —  idénticos al Problema 3
# ---------------------------------------------------------------------------
def overlap(S, patron=None):
    p = xi if patron is None else patron
    return float(np.dot(S, p)) / len(p)

def corrompe(patron, n, rng):
    """Invierte EXACTAMENTE n píxeles elegidos al azar."""
    S = patron.copy()
    idx = rng.choice(len(patron), size=int(n), replace=False)
    S[idx] *= -1
    return S

# ---------------------------------------------------------------------------
# 4. Dinámica asíncrona: un píxel al azar por paso (criterio h_i > 0 -> +1)
# ---------------------------------------------------------------------------
def traza(W, S0, pasos, rng):
    S = S0.copy()
    ms = [overlap(S)]
    for _ in range(pasos):
        i = rng.integers(N)
        S[i] = 1 if W[i] @ S > 0 else -1
        ms.append(overlap(S))
    return ms

def evoluciona(W, S0, barridos, rng):
    S = S0.copy()
    for _ in range(barridos):
        for i in rng.permutation(N):
            S[i] = 1 if W[i] @ S > 0 else -1
    return S

# ===========================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # EXP 2 — Overlap vs paso para distintos K
    # -----------------------------------------------------------------------
    n = 47
    pasos = 600
    R = 40
    Ks = [99, 50, 30, 15, 8]

    # Se calculan primero todas las trazas y luego se dibujan de la banda
    # MÁS ANCHA (K pequeño) a la MÁS ESTRECHA (K grande), para que las bandas
    # estrechas no queden tapadas por las anchas dibujadas después.
    # Además se traza el contorno (m-s, m+s) con línea discontinua del mismo
    # color, para poder seguir el borde de cada banda aunque los rellenos
    # se solapen y se mezclen visualmente.
    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    curvas = {}
    for K, color in zip(Ks, colores):
        T = np.array([traza(W_diluida(K, rng), corrompe(xi, n, rng), pasos, rng)
                      for _ in range(R)])
        curvas[K] = (T.mean(0), T.std(0), color)

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for K in sorted(Ks):  # K ascendente = banda descendente en anchura (más ancha primero)
        m, s, color = curvas[K]
        pasos_x = range(len(m))
        ax.fill_between(pasos_x, m - s, m + s, color=color, alpha=.10, zorder=1)
        ax.plot(pasos_x, m - s, color=color, lw=.6, ls=":", alpha=.8, zorder=2)
        ax.plot(pasos_x, m + s, color=color, lw=.6, ls=":", alpha=.8, zorder=2)
    for K in Ks:
        m, s, color = curvas[K]
        ax.plot(m, lw=1.6, color=color, label=f"K = {K}  (w = {K/N:.2f})", zorder=3)
    ax.set(xlabel="paso", ylabel="overlap  m",
           title=f"Overlap vs paso para distintos K   (n = {n})")
    ax.axhline(1, ls="--", c="gray", lw=.7); ax.axhline(0, ls=":", c="gray", lw=.6)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("p4_exp2_overlap_vs_K.png", dpi=140); plt.close(fig)

    # -----------------------------------------------------------------------
    # EXP 3 — No determinismo inducido por dilución
    # -----------------------------------------------------------------------
    n2, K2, corridas = 48, 15, 8
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for _ in range(corridas):
        ax.plot(traza(J, corrompe(xi, n2, rng), pasos, rng), c="#185FA5", lw=1, alpha=.7)
    for _ in range(corridas):
        ax.plot(traza(W_diluida(K2, rng), corrompe(xi, n2, rng), pasos, rng),
                c="#A32D2D", lw=1, alpha=.7)
    ax.plot([], [], c="#185FA5", label="red completa  K = 99")
    ax.plot([], [], c="#A32D2D", label=f"diluida  K = {K2}")
    ax.set(xlabel="paso", ylabel="overlap  m",
           title=f"No determinismo inducido por dilución   (n = {n2})")
    ax.axhline(1, ls="--", c="gray", lw=.7); ax.axhline(0, ls=":", c="gray", lw=.6)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("p4_exp3_nodeterminismo.png", dpi=140); plt.close(fig)

    # -----------------------------------------------------------------------
    # EXP 4 — Calidad de recuerdo m_c vs fracción destruida
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    K_lista = [99, 80, 60, 45, 30, 20, 12, 8, 5, 3, 2, 1]
    for n_pix in [10, 25, 40]:
        destr, medias = [], []
        for K in K_lista:
            vals = [traza(W_diluida(K, rng), corrompe(xi, n_pix, rng), pasos, rng)[-1]
                    for _ in range(30)]
            destr.append(1 - K/N); medias.append(np.mean(vals))
        ax.plot(destr, medias, marker="o", ms=4, label=f"n = {n_pix} píxeles")
    ax.set(xlabel="fracción destruida  (1 - w),  w = K/N",
           ylabel="calidad de recuerdo  m_c",
           title="Recuerdo vs dilución (grado fijo, un patrón)")
    ax.axhline(1, ls="--", c="gray", lw=.7); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("p4_exp4_recuerdo.png", dpi=140); plt.close(fig)

    # -----------------------------------------------------------------------
    # EXP 5 — Verificación del análisis señal/ruido del campo local h_i(0)
    #   Señal:  <h_i(0)/xi_i> = w (1 - 2n/N)                      (exacta)
    #   Ruido:  desv. típica de sum_j D_ij sigma_j
    #             = 2 sqrt( K (n/N)(1-n/N) )  ~  sqrt(K)
    #   Esta fórmula del ruido asume los K sumandos independientes, aproximación
    #   válida en el régimen diluido K << N (que es el que se verifica aquí).
    # -----------------------------------------------------------------------
    def campo_sobre_xi(K, n, rng):
        """Devuelve h_i(0)/xi_i para todas las neuronas, con n píxeles corruptos.
           h_i(0)/xi_i = (1/N) sum_j D_ij sigma_j."""
        D = mascara_regular(N, K, rng) if K < N - 1 else (np.ones((N, N)) - np.eye(N))
        S = corrompe(xi, n, rng)
        return ((D * J) @ S) / xi

    n_v = 30                                   # corrupción fija para la verificación
    f_v = n_v / N
    # Barrido desde el régimen diluido K << N hacia K mayores. Se incluyen K
    # altos a propósito para ver cómo la simulación se separa de la ley simple
    # sqrt(K), válida solo en K << N (el muestreo sin reemplazo reduce el ruido
    # al crecer K/N).
    Ks_v = np.array([2, 4, 6, 8, 10, 12, 16, 20, 25, 30,
                     36, 42, 48, 54, 60])

    senal_sim, ruido_sim = [], []
    for K in Ks_v:
        # señal: media de h_i(0)/xi_i  ->  w(1-2n/N)
        muestras_h = np.array([campo_sobre_xi(K, n_v, rng) for _ in range(300)])
        senal_sim.append(muestras_h.mean())
        # ruido: desv. típica de la SUMA  sum_j D_ij sigma_j = N * (h/xi)
        muestras_S = np.array([campo_sobre_xi(K, n_v, rng) * N for _ in range(600)])
        ruido_sim.append(muestras_S.std())

    # --- ajustes de tendencia (mínimos cuadrados por el origen, régimen K<=12) ---
    senal_sim = np.array(senal_sim); ruido_sim = np.array(ruido_sim)
    val = Ks_v <= 12
    a_fit = np.sum(senal_sim[val] * Ks_v[val]) / np.sum(Ks_v[val] ** 2)          # señal ~ a K
    c_fit = np.sum(ruido_sim[val] * np.sqrt(Ks_v[val])) / np.sum(Ks_v[val])      # ruido ~ c sqrt(K)
    Kf = np.linspace(0, Ks_v.max() * 1.05, 200)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(Kf, a_fit * Kf, "k--", lw=1, label=r"tendencia $\propto K$")
    a1.plot(Ks_v, senal_sim, "o", ms=6, label="simulación")
    a1.set(xlabel="K", ylabel=r"señal  $\langle h_i(0)/\xi_i\rangle$",
           title=r"Señal $\propto K$")
    a1.legend(); a1.grid(alpha=.3)

    # ley simple sqrt(K), ajustada SOLO en K<<N (representa la aproximación pura)
    a2.plot(Kf, c_fit * np.sqrt(Kf), "k--", lw=1,
            label=r"$\propto\sqrt{K}$")
    a2.plot(Ks_v, ruido_sim, "s", ms=6, color="#0F6E56", label="simulación")
    a2.set(xlabel="K", ylabel=r"ruido  (desv. típica de $\sum_j D_{ij}\sigma_j$)",
           title=r"Ruido: $\sqrt{K}$")
    a2.legend(fontsize=8); a2.grid(alpha=.3)
    fig.suptitle(r"Leyes de escala señal/ruido  ($n = %d$)"
                 % n_v,
                 fontsize=12)
    fig.tight_layout(); fig.savefig("p4_exp5_senal_ruido.png", dpi=140); plt.close(fig)
