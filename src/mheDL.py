import numpy as np
from numpy import random
from scipy.linalg import expm, solve, inv, cholesky
from scipy.signal import place_poles
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# --- System definition (continuous) ---
Acont = np.array([
    [-0.026, 0.074, -0.804, -9.809, 0],
    [-0.242, -2.017, 73.297, -0.105, -0.001],
    [0.003, -0.135, -2.941, 0, 0],
    [0, 0, 1, 0, 0],
    [-0.011, 1, 0, -75, 0]
])
Bcont = np.array([
    [4.594, 0],
    [-0.0004, -13.735],
    [0.0002, -24.410],
    [0, 0],
    [0, 0]
])
C = np.array([
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])

Delta = 0.1
Nx, Nu, Ny = 5, 2, 3
Nsim = 50
t = np.arange(Nsim+1) * Delta

# Discretize
A = expm(Acont * Delta)
B = np.linalg.solve(Acont, (A - np.eye(Nx))).dot(Bcont)

Q = 0.01 * np.eye(Nx)
R = 0.1 * np.eye(Ny)

# MHE
class LinearMHE:
    def __init__(self, A, B, C, Q, R, P0inv):
        self.A = A
        self.B = B
        self.C = C
        self.Qinv = inv(Q)
        self.Rinv = inv(R)
        self.Pinv = P0inv
        self.nx = A.shape[0]

    def estimate(self, x0bar, u_seq, y_seq):
        M = y_seq.shape[0] - 1
        n = self.nx
        Nvar = (M+1)*n
        H = np.zeros((Nvar, Nvar))
        g = np.zeros(Nvar)

        H[0:n, 0:n] += self.Pinv
        g[0:n] += self.Pinv @ x0bar

        for k in range(M+1):
            i = k*n
            H[i:i+n, i:i+n] += self.C.T @ self.Rinv @ self.C
            g[i:i+n] += self.C.T @ self.Rinv @ y_seq[k]

        for k in range(M):
            i = k*n; j = (k+1)*n
            H[i:i+n, i:i+n] += self.A.T @ self.Qinv @ self.A
            H[i:i+n, j:j+n] += -self.A.T @ self.Qinv
            H[j:j+n, i:i+n] += -self.Qinv @ self.A
            H[j:j+n, j:j+n] += self.Qinv
            bu = self.B @ u_seq[k]
            g[i:i+n] += -self.A.T @ self.Qinv @ bu
            g[j:j+n] +=  self.Qinv @ bu

        X = solve(H, g)
        return X[-n:]

# Generate dataset
random.seed(42)
num_simulations = 1000
horizon = 25
omega = 2*np.pi/(Nsim*Delta)

X_mhe_vanilla = []
Y_mhe_vanilla = []

mhe_vanilla = LinearMHE(A, B, C, Q, R, inv(np.eye(Nx)))

for sim in range(num_simulations):
    x_sim = np.zeros((Nsim+1, Nx))
    x_sim[0] = np.random.uniform(-1, 1, size=Nx)
    u_sim = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
    y_sim = np.zeros((Nsim+1, Ny))

    w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
    v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)

    for k in range(Nsim):
        x_sim[k+1] = A @ x_sim[k] + B @ u_sim[k] + w[:, k]
        y_sim[k] = C @ x_sim[k] + v[:, k]
    y_sim[Nsim] = C @ x_sim[Nsim] + v[:, Nsim]

    xhat_sim = np.zeros((Nsim+1, Nx))

    for k in range(Nsim):
        tmin = max(0, k-horizon)
        tmax = k+1
        u_win = u_sim[tmin:tmax-1] if k >= 1 else np.zeros((0, Nu))
        y_win = y_sim[tmin:tmax]
        x0bar = xhat_sim[k].copy()
        xhat_sim[k+1] = mhe_vanilla.estimate(x0bar, u_win, y_win)

        X_mhe_vanilla.append(np.hstack([xhat_sim[k], u_sim[k], y_sim[k+1]]))
        Y_mhe_vanilla.append(xhat_sim[k+1])

X_mhe_data = np.array(X_mhe_vanilla)
Y_mhe_data = np.array(Y_mhe_vanilla)

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X_mhe_data, Y_mhe_data, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

class MHE_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = MHE_Net(X_train.shape[1], 64, Nx)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train_model(model, loader, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader):.4f}")

train_model(model, train_loader, optimizer, criterion)


import numpy as np
from scipy.linalg import cholesky, expm
import matplotlib.pyplot as plt
import torch
import time

# --- Constants ---
Nx, Nu, Ny = 5, 2, 3
Nsim = 50
Delta = 0.1
t = np.arange(Nsim+1) * Delta
omega = 2 * np.pi / (Nsim * Delta)

# --- Simulate one trajectory ---
A = expm(Acont * Delta)
B = np.linalg.solve(Acont, (A - np.eye(Nx))).dot(Bcont)
Q = 0.01 * np.eye(Nx)
R = 0.1 * np.eye(Ny)

x_true = np.zeros((Nsim+1, Nx))
y_meas = np.zeros((Nsim+1, Ny))
u = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
x_true[0] = np.zeros(Nx)

w = cholesky(Q, lower=True) @ np.random.randn(Nx, Nsim)
v = cholesky(R, lower=True) @ np.random.randn(Ny, Nsim+1)

for k in range(Nsim):
    x_true[k+1] = A @ x_true[k] + B @ u[k] + w[:, k]
    y_meas[k] = C @ x_true[k] + v[:, k]
y_meas[Nsim] = C @ x_true[Nsim] + v[:, Nsim]

# --- Vanilla MHE ---
xhat_mhe = np.zeros((Nsim+1, Nx))
start_time_mhe = time.perf_counter()
for k in range(Nsim):
    tmin = max(0, k - horizon)
    tmax = k + 1
    u_win = u[tmin:tmax-1] if k >= 1 else np.zeros((0, Nu))
    y_win = y_meas[tmin:tmax]
    x0bar = xhat_mhe[k].copy()
    xhat_mhe[k+1] = mhe_vanilla.estimate(x0bar, u_win, y_win)
end_time_mhe = time.perf_counter()

# --- Neural Network Estimator --- # Remettre en forme fonction de cout
xhat_nn = np.zeros((Nsim+1, Nx))
xhat_nn[0] = np.zeros(Nx)
start_time_nn = time.perf_counter()
for k in range(Nsim):
    x_input = np.hstack([xhat_nn[k], u[k], y_meas[k+1]])
    x_input_scaled = scaler_X.transform(x_input.reshape(1, -1))
    x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)
    with torch.no_grad():
        x_next_scaled = model(x_input_tensor).numpy()
    xhat_nn[k+1] = scaler_y.inverse_transform(x_next_scaled)
end_time_nn = time.perf_counter()

# --- Timing ---
avg_time_mhe = (end_time_mhe - start_time_mhe) / Nsim * 1e3
avg_time_nn = (end_time_nn - start_time_nn) / Nsim * 1e3

print(f"[MHE] Avg. time per iter: {avg_time_mhe:.3f} ms")
print(f"[NN ] Avg. time per iter: {avg_time_nn:.3f} ms")

# --- Plot ---
plt.figure(figsize=(10, Nx*2))
for i in range(Nx):
    plt.subplot(Nx, 1, i+1)
    plt.plot(t, x_true[:, i], label='True')
    plt.plot(t, xhat_mhe[:, i], '--', label='MHE')
    plt.plot(t, xhat_nn[:, i], ':', label='NN')
    plt.ylabel(f'$x_{i}$')
    plt.legend()
plt.xlabel("Time [s]")
plt.suptitle("State Estimation: Vanilla MHE vs NN")
plt.tight_layout()
plt.show()

times_mhe = []
for k in range(Nsim):
    start = time.perf_counter()

    tmin = max(0, k - horizon)
    tmax = k + 1
    u_win = u[tmin:tmax-1] if k >= 1 else np.zeros((0, Nu))
    y_win = y_meas[tmin:tmax]
    x0bar = xhat_mhe[k].copy()
    xhat_mhe[k+1] = mhe_vanilla.estimate(x0bar, u_win, y_win)

    end = time.perf_counter()
    times_mhe.append((end - start) * 1e3)  # milliseconds

times_nn = []
for k in range(Nsim):
    start = time.perf_counter()

    x_input = np.hstack([xhat_nn[k], u[k], y_meas[k+1]])
    x_input_scaled = scaler_X.transform(x_input.reshape(1, -1))
    x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        x_next_scaled = model(x_input_tensor).numpy()
    xhat_nn[k+1] = scaler_y.inverse_transform(x_next_scaled)

    end = time.perf_counter()
    times_nn.append((end - start) * 1e3)  # milliseconds

plt.figure(figsize=(10, 5))
plt.plot(times_mhe, label="MHE time per step")
plt.plot(times_nn, label="NN time per step")
plt.xlabel("Iteration (k)")
plt.ylabel("Time [ms]")
plt.title("Computation Time per Estimation Step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

