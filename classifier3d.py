import numpy as np

# ------------------------------------------------------------
# 1. Charger les données
# Le CSV doit avoir les colonnes : x, y, z, label
# et une ligne d'en-tête.
# ------------------------------------------------------------

data = np.genfromtxt(
    "points.csv",
    delimiter=",",
    skip_header=1,
    dtype=str
)

X = data[:, :3].astype(np.float32)  # coordonnées x, y, z
labels = data[:, 3]                 # classe de chaque point

# Transforme les étiquettes en nombres : classe 0, classe 1, etc.
classes, y = np.unique(labels, return_inverse=True)
y = y.astype(np.int64)

if X.ndim != 2 or X.shape[1] != 3:
    raise ValueError("X doit avoir exactement 3 colonnes : x, y et z.")
if len(X) != len(y):
    raise ValueError("Il faut une étiquette par point.")

# ------------------------------------------------------------
# 2. Séparer les données : entraînement, validation et test
# ------------------------------------------------------------

rng = np.random.default_rng(42)
indices = rng.permutation(len(X))

n_train = int(0.8 * len(X))
n_val = int(0.1 * len(X))

i_train = indices[:n_train]
i_val = indices[n_train:n_train + n_val]
i_test = indices[n_train + n_val:]

X_train, y_train = X[i_train], y[i_train]
X_val, y_val = X[i_val], y[i_val]
X_test, y_test = X[i_test], y[i_test]

# Normalisation calculée uniquement sur les données d'entraînement
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
sd[sd < 1e-8] = 1.0

X_train = (X_train - mu) / sd
X_val = (X_val - mu) / sd
X_test = (X_test - mu) / sd

# ------------------------------------------------------------
# 3. Construire le réseau : 3 entrées -> 32 neurones -> classes
# ------------------------------------------------------------

n_features = X_train.shape[1]  # 3 coordonnées
n_classes = len(classes)
n_hidden = 32

params = {
    "W1": rng.normal(
        0, np.sqrt(2 / n_features), size=(n_features, n_hidden)
    ).astype(np.float32),
    "b1": np.zeros(n_hidden, dtype=np.float32),
    "W2": rng.normal(
        0, np.sqrt(2 / n_hidden), size=(n_hidden, n_classes)
    ).astype(np.float32),
    "b2": np.zeros(n_classes, dtype=np.float32),
}

def forward(x, p):
    z1 = x @ p["W1"] + p["b1"]
    h1 = np.maximum(0, z1)  # ReLU

    logits = h1 @ p["W2"] + p["b2"]
    logits -= logits.max(axis=1, keepdims=True)  # stabilité numérique

    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return z1, h1, probs

def loss_and_accuracy(x, y_true, p):
    _, _, probs = forward(x, p)
    loss = -np.log(probs[np.arange(len(y_true)), y_true] + 1e-12).mean()
    accuracy = (probs.argmax(axis=1) == y_true).mean()
    return loss, accuracy

# ------------------------------------------------------------
# 4. Entraîner le réseau avec Adam et mini-lots
# ------------------------------------------------------------

learning_rate = 0.001
batch_size = 256
max_epochs = 150

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
step = 0

m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(val) for k, val in params.items()}

best_val_loss = float("inf")
best_params = {k: val.copy() for k, val in params.items()}
patience = 15
epochs_without_improvement = 0

for epoch in range(max_epochs):
    shuffled = rng.permutation(len(X_train))

    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start:start + batch_size]
        xb = X_train[batch]
        yb = y_train[batch]
        n = len(xb)

        z1, h1, probs = forward(xb, params)

        # Gradient de la cross-entropy avec softmax
        dz2 = probs.copy()
        dz2[np.arange(n), yb] -= 1
        dz2 /= n

        grads = {}
        grads["W2"] = h1.T @ dz2
        grads["b2"] = dz2.sum(axis=0)

        dh1 = dz2 @ params["W2"].T
        dz1 = dh1 * (z1 > 0)

        grads["W1"] = xb.T @ dz1
        grads["b1"] = dz1.sum(axis=0)

        # Mise à jour Adam
        step += 1

        for key in params:
            m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
            v[key] = beta2 * v[key] + (1 - beta2) * grads[key] ** 2

            m_hat = m[key] / (1 - beta1 ** step)
            v_hat = v[key] / (1 - beta2 ** step)

            params[key] -= learning_rate * m_hat / (
                np.sqrt(v_hat) + epsilon
            )

    val_loss, val_accuracy = loss_and_accuracy(X_val, y_val, params)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {k: val.copy() for k, val in params.items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(
            f"Époque {epoch + 1:3d} | "
            f"perte validation : {val_loss:.4f} | "
            f"précision : {val_accuracy:.3f}"
        )

    # Arrête si les résultats de validation ne s'améliorent plus
    if epochs_without_improvement >= patience:
        print("Arrêt anticipé.")
        break

# ------------------------------------------------------------
# 5. Évaluer sur les points réservés au test
# ------------------------------------------------------------

test_loss, test_accuracy = loss_and_accuracy(X_test, y_test, best_params)

print(f"\nPerte sur le test : {test_loss:.4f}")
print(f"Précision sur le test : {test_accuracy:.3f}")
print("Classes reconnues :", classes)

# ------------------------------------------------------------
# 6. Prédire la classe de nouveaux points [x, y, z]
# ------------------------------------------------------------

def predict(points_3d):
    points_3d = np.asarray(points_3d, dtype=np.float32)

    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, -1)

    if points_3d.shape[1] != 3:
        raise ValueError("Chaque point doit avoir 3 coordonnées : x, y, z.")

    points_normalized = (points_3d - mu) / sd
    _, _, probs = forward(points_normalized, best_params)

    indices_predits = probs.argmax(axis=1)
    classes_predites = classes[indices_predits]

    return classes_predites

# Exemple :
# print(predict([[0.2, 1.5, -0.3], [0.4, 1.2, -0.1]]))
