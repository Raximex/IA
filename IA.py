import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hidden(X, W_1, b):
    return sigmoid(np.dot(W_1, X) + b)

def output(H, W_2, b):
    return sigmoid(np.dot(W_2.T, H) + b)

def delta_2(o, y):
    return float((o - y) * o * (1 - o))

def delta_1(W_2, h, d2):
    d1 = np.zeros(shape=(3, 1))
    for j in range(d1.shape[0]):
        d1[j] = W_2[j] * d2 * h[j] * (1 - h[j])
    return d1

def dJdW_2(h, d2):
    dj = np.zeros(shape=(3, 1))
    for j in range(dj.shape[0]):
        dj[j] = d2 * h[j]
    return dj

def dJdW_1(X, d1):
    dj = np.zeros(shape=(3, 2))
    for j in range(dj.shape[0]):
        for i in range(dj.shape[1]):
            dj[j, i] = d1[j] * X[j, i]
    return dj

def train_model(X, W_1, W_2, b_1, b_2, y, alpha):
    for i in range(y.shape[1]):
        h = hidden(np.reshape(X[i], (-1,1)), W_1, b_1)
        o = output(h, W_2, b_2)
        d2, djdb_2 = delta_2(o, y[i]), delta_2(o, y[i])
        d1, djdb_1 = delta_1(W_2, h, d2), delta_1(W_2, h, d2)
        dj2 = dJdW_2(h, d2)
        dj1 = dJdW_1(X[i].reshape(-1, 1), d1)
        W_2 -= alpha * dj2
        W_1 -= alpha * dj1
        b_1 -= alpha * djdb_1
        b_2 -= alpha * djdb_2
    return W_1, W_2, b_1, b_2

def predict_model(X, W_1, W_2, b_1, b_2, y):
    y_hat = []
    for i in range(X.shape(0)):
        h = hidden(X[i].reshape(-1, 1), W_1, b_1)
        o = output(h, W_2, b_2)
        y_hat.append(float(o))
    return y_hat

# Définir les données d'entrée et de sortie
np.random.seed(42)
X = np.random.standard_normal(size=(2, 1))
W_1 = np.random.standard_normal(size=(3, 2))
W_2 = np.random.standard_normal(size=(3, 1))
b_1 = np.random.standard_normal(size=(3, 1))
b_2 = np.random.standard_normal(size=(1, 1))
y = np.random.standard_normal(size=(1, 1))

# Entraînement du modèle
W_1_trained, W_2_trained, b_1_trained, b_2_trained = train_model(X, W_1, W_2, b_1, b_2, y, 0.01)

# Prédiction du modèle
predictions = predict_model(X, W_1_trained, W_2_trained, b_1_trained, b_2_trained, y)
print(predictions)
