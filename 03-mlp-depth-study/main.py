"""
Problema 7B - Efecto de la profundidad en una red feed-forward entrenada con backpropagation
Dataset: South German Credit (Statlog)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# 1. CARGA DE DATOS
# ---------------------------------------------------------------------------
URL = "https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv"
df = pd.read_csv(URL)

print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Balance de clases (credit_risk): \n{df['credit_risk'].value_counts()}\n")

# ---------------------------------------------------------------------------
# 2. PREPROCESADO
# ---------------------------------------------------------------------------
y = df["credit_risk"].values
X_raw = df.drop(columns=["credit_risk"])

# One-hot encoding de las variables categóricas (status, purpose, savings, etc.)
X = pd.get_dummies(X_raw, drop_first=True)

# Split train/test estratificado (por el desbalance 70/30 de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# Escalado
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dimensión de entrada tras one-hot encoding: {X_train.shape[1]} variables")
print(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras\n")

# ---------------------------------------------------------------------------
# 3. BARRIDO DE PROFUNDIDAD
# ---------------------------------------------------------------------------
DEPTHS = [1, 2, 3, 4, 6, 8]
WIDTH = 16          # nº de neuronas por capa oculta, fijo para aislar el efecto de la profundidad
LR = 0.05            # eta, fijo para todas las configuraciones
MAX_ITER = 4000

results = {}

for depth in DEPTHS:
    hidden_layers = tuple([WIDTH] * depth)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="logistic",   # sigmoide
        solver="sgd",
        momentum=0.0,            # sin momentum
        learning_rate_init=LR,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        tol=1e-6,
        n_iter_no_change=MAX_ITER,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    results[depth] = {
        "loss_curve": clf.loss_curve_,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_iter": clf.n_iter_,
        "final_loss": clf.loss_curve_[-1],
        "time": elapsed,
    }

    print(f"Profundidad {depth} capa(s) oculta(s): "
          f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
          f"loss_final={clf.loss_curve_[-1]:.4f}  iters={clf.n_iter_}  ({elapsed:.1f}s)")

# ---------------------------------------------------------------------------
# 4. GRÁFICAS (dos figuras independientes)
# ---------------------------------------------------------------------------

# --- Figura 1: Accuracy vs. profundidad ---
fig1, ax1 = plt.subplots(figsize=(7, 5))
train_accs = [results[d]["train_acc"] for d in DEPTHS]
test_accs = [results[d]["test_acc"] for d in DEPTHS]
ax1.plot(DEPTHS, train_accs, "o-", label="Train Precision")
ax1.plot(DEPTHS, test_accs, "s-", label="Test Precision")
ax1.set_xlabel("Número de capas ocultas")
ax1.set_ylabel("Precision")
ax1.set_title("Precision vs. profundidad de la red")
ax1.set_xticks(DEPTHS)
ax1.legend()
ax1.grid(alpha=0.3)
fig1.tight_layout()
fig1.savefig("resultados_7b_accuracy.png", dpi=150)

# --- Figura 2: Curvas de convergencia por profundidad ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
for d in DEPTHS:
    ax2.plot(results[d]["loss_curve"], label=f"{d} capa(s)")
ax2.set_xlabel("Iteracion")
ax2.set_ylabel("Error (log-loss)")
ax2.set_title("Curvas de convergencia por profundidad")
ax2.set_yscale("log")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("resultados_7b_convergencia.png", dpi=150)

