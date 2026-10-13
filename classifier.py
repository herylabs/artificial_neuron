import numpy as np

# X : tableau (n_points, 2), par exemple [[x1, y1], [x2, y2], ...]
# y : tableau (n_points,), par exemple [0, 1, 0, 2, ...]
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y).reshape(-1)

# Encode les classes en 0, 1, 2, ... (fonctionne aussi avec des étiquettes texte)
classes, y = np.unique(y, return_inverse=True)
y = y.astype(np.int64)

# Sépare les données : 80 % entraînement, 10 % validation, 10 % test
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

# Normalise les coordonnées à partir du jeu d'entraînement uniquement
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
sd[sd < 1e-8] = 1.0

X_train = (X_train - mu) / sd
X_val = (X_val - mu) / sd
X_test = (X_test - mu) / sd

# Réseau : 2 entrées -> 32 neurones ReLU -> une sortie par classe
n_classes = len(classes)
params = {
    "W1": rng.normal(0, np.sqrt(2 / 2), size=(2, 32)).astype(np.float32),
    "b1": np.zeros(32, dtype=np.float32),
    "W2": rng.normal(0, np.sqrt(2 / 32), size=(32, n_classes)).astype(np.float32),
    "b2": np.zeros(n_classes, dtype=np.float32),
}

def forward(x, p):
    z1 = x @ p["W1"] + p["b1"]
    h1 = np.maximum(0, z1)
    logits = h1 @ p["W2"] + p["b2"]
    logits -= logits.max(axis=1, keepdims=True)  # stabilité numérique
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return z1, h1, probs

def loss_and_accuracy(x, y, p):
    _, _, probs = forward(x, p)
    loss = -np.log(probs[np.arange(len(y)), y] + 1e-12).mean()
    accuracy = (probs.argmax(axis=1) == y).mean()
    return loss, accuracy

# Adam
m = {k: np.zeros_like(v) for k, v in params.items()}
v = {k: np.zeros_like(val) for k, val in params.items()}
beta1, beta2, eps = 0.9, 0.999, 1e-8
learning_rate = 0.001
step = 0

best_val_loss = float("inf")
best_params = {k: val.copy() for k, val in params.items()}
patience = 15
epochs_without_improvement = 0

# Entraînement
batch_size = 256
max_epochs = 150

for epoch in range(max_epochs):
    shuffled = rng.permutation(len(X_train))

    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start:start + batch_size]
        xb, yb = X_train[batch], y_train[batch]
        n = len(xb)

        z1, h1, probs = forward(xb, params)

        # Gradient de la cross-entropy + softmax
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
            v[key] = beta2 * v[key] + (1 - beta2) * grads[key]**2

            m_hat = m[key] / (1 - beta1**step)
            v_hat = v[key] / (1 - beta2**step)
            params[key] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    val_loss, val_acc = loss_and_accuracy(X_val, y_val, params)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {k: val.copy() for k, val in params.items()}
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if (epoch + 1) % 10 == 0:
        print(f"Époque {epoch + 1:3d} | perte val. {val_loss:.4f} | précision val. {val_acc:.3f}")

    if epochs_without_improvement >= patience:
        print("Arrêt anticipé.")
        break

# Résultat final sur les points jamais utilisés pendant l'entraînement
test_loss, test_acc = loss_and_accuracy(X_test, y_test, best_params)
print(f"Test | perte {test_loss:.4f} | précision {test_acc:.3f}")

def predict(X_new):
    X_new = np.asarray(X_new, dtype=np.float32)
    X_new = (X_new - mu) / sd
    _, _, probs = forward(X_new, best_params)
    return classes[probs.argmax(axis=1)]
