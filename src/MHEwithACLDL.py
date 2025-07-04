import numpy as np
from numpy import random
from scipy.linalg import expm, solve, inv, cholesky
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import time

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

# Sampling
Delta = 0.1
Nx, Nu, Ny = 5, 2, 3
Nsim = 50
t = np.arange(Nsim+1) * Delta

# Discretize
A = expm(Acont*Delta)
# exact B discretization:

from scipy.linalg import solve_continuous_lyapunov
# simple zero‐order hold:
B = np.linalg.solve(Acont, (A - np.eye(Nx))).dot(Bcont)

# Noise covariances
Q = 0.01 * np.eye(Nx)
R = 0.1  * np.eye(Ny)

# Observer gain L (A-CL)
desired_poles = [0.6, 0.61, 0.62, 0.63, 0.64]
L = place_poles(A.T, C.T, desired_poles).gain_matrix.T

# Analytic MHE + pre-estimator
class LinearMHEPreEstimator:
    def __init__(self, A, B, C, Q, R, Pinv):
        self.A = A; self.B = B; self.C = C
        self.Qinv = inv(Q); self.Rinv = inv(R); self.Pinv = Pinv
        self.nx = A.shape[0]

    def estimate(self, x0bar, u_seq, y_seq):
        M = y_seq.shape[0] - 1
        n = self.nx
        Nvar = (M+1)*n
        H = np.zeros((Nvar, Nvar))
        g = np.zeros(Nvar)

        # Initial‐state prior
        H[0:n,0:n] += self.Pinv
        g[0:n]     += self.Pinv @ x0bar

        # Measurement terms
        for k in range(M+1):
            i = k*n
            H[i:i+n, i:i+n] += self.C.T @ self.Rinv @ self.C
            g[i:i+n]       += self.C.T @ self.Rinv @ y_seq[k]
        # Dynamics terms
        for k in range(M):
            i = k*n; j = (k+1)*n
            H[i:i+n, i:i+n] += self.A.T @ self.Qinv @ self.A
            H[i:i+n, j:j+n] += -self.A.T @ self.Qinv
            H[j:j+n, i:i+n] += -self.Qinv @ self.A
            H[j:j+n, j:j+n] += self.Qinv
            bu = self.B @ u_seq[k]
            g[i:i+n] += -self.A.T @ self.Qinv @ bu
            g[j:j+n] +=  self.Qinv @ bu
        # Solve
        X = solve(H, g)
        return X[-n:]

# Precompute inverses
Qinv = inv(Q)
Rinv = inv(R)
P0 = np.eye(Nx)     
Pinv0 = inv(P0)

# Build estimator 
horizon = 25
mhe = LinearMHEPreEstimator(A, B, C, Q, R, Pinv0) 

random.seed(927)
x_true = np.zeros((Nsim+1, Nx))
y_meas = np.zeros((Nsim+1, Ny))
x_true[0] = np.zeros(Nx)
omega = 2*np.pi/(Nsim*Delta)
u = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)
for k in range(Nsim+1):
    y_meas[k] = C @ x_true[k] + v[:,k]
    if k<Nsim:
        x_true[k+1] = A @ x_true[k] + B @ u[k] + w[:,k]

xhat = np.zeros((Nsim+1, Nx))
xhat[0] = np.zeros(Nx)
x0bar = xhat[0].copy()
P = P0.copy()

times = []
total_start_time = time.perf_counter()

for k in range(Nsim):
    t0 = time.perf_counter()

    # Build window
    M = min(max(k,1), horizon)
    tmin = max(0, k-horizon)
    tmax = k+1
    u_win = u[tmin:tmax-1] if k>=1 else np.zeros((0, Nu))
    y_win = y_meas[tmin:tmax]

    # Pre‐estimator update (A-CL)
    if k>0:
        x0bar = A @ xhat[k] + B @ u[k-1] + L @ (y_meas[k] - C @ xhat[k])
        P = (A - L@C)@P@(A - L@C).T + Q + L@R@L.T
        mhe.Pinv = inv(P)

    # MHE analytic solve
    xhat[k+1] = mhe.estimate(x0bar, u_win, y_win)

    times.append(time.perf_counter() - t0)
total_end_time = time.perf_counter()

# --- Results ---
print(f"Avg. time per iter: {np.mean(times)*1e3:.6f} ms")
print(f"Total time for {Nsim} iterations: {(total_end_time - total_start_time)*1e3:.6f} ms")

# Plots
fig, axs = plt.subplots(Nx,1, figsize=(8,Nx*2))
for i in range(Nx):
    axs[i].plot(t, x_true[:,i], label=f"True $x_{i}$")
    axs[i].plot(t, xhat[:,i],'--', label=f"Est. $x_{i}$")
    axs[i].legend(); axs[i].set_ylabel(f"$x_{i}$")
axs[-1].set_xlabel("Time [s]")
plt.suptitle("Analytic Linear MHE + A-CL")
plt.tight_layout()
plt.show() 


# Monte Carlo dataset generation from MHE
num_simulations = 1000
horizon = 25
seq_length = 1  # Only 1-step MHE output is used

X_mhe_data = []
Y_mhe_data = []

for sim in range(num_simulations):
    x_sim = np.zeros((Nsim+1, Nx))
    x_sim[0] = np.random.uniform(-1, 1, size=Nx)
    u_sim = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
    y_sim = np.zeros((Nsim+1, Ny))

    # Add noise
    w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
    v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)

    for k in range(Nsim):
        x_sim[k+1] = A @ x_sim[k] + B @ u_sim[k] + w[:, k]
        y_sim[k] = C @ x_sim[k] + v[:, k]
    y_sim[Nsim] = C @ x_sim[Nsim] + v[:, Nsim]

    xhat_sim = np.zeros((Nsim+1, Nx))
    x0bar = x_sim[0].copy()
    P = np.eye(Nx)
    
    for k in range(Nsim):
        tmin = max(0, k-horizon)
        tmax = k+1
        u_win = u_sim[tmin:tmax-1] if k >= 1 else np.zeros((0, Nu))
        y_win = y_sim[tmin:tmax]

        if k > 0:
            x0bar = A @ xhat_sim[k] + B @ u_sim[k-1] + L @ (y_sim[k] - C @ xhat_sim[k])
            P = (A - L @ C) @ P @ (A - L @ C).T + Q + L @ R @ L.T
            mhe.Pinv = inv(P)
        
        xhat_sim[k+1] = mhe.estimate(x0bar, u_win, y_win)

        # Save features and labels
        X_mhe_data.append(np.hstack([xhat_sim[k], u_sim[k], y_sim[k+1]]))  # input
        Y_mhe_data.append(xhat_sim[k+1])  # target

X_mhe_data = np.array(X_mhe_data)
Y_mhe_data = np.array(Y_mhe_data)

print("MHE dataset shapes:", X_mhe_data.shape, Y_mhe_data.shape)
# Data preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_mhe_data, Y_mhe_data, test_size=0.2, random_state=42)

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define NN
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

# Training loop
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

# Run estimation using NN
xhat_nn = np.zeros((Nsim+1, Nx))
xhat_nn[0] = np.zeros(Nx)
start_time_nn = time.perf_counter()

for k in range(Nsim):
    
    x_input = np.hstack([xhat_nn[k], u[k], y_meas[k+1]])
    x_input_scaled = scaler_X.transform(x_input.reshape(1, -1))
    x_input_tensor = torch.tensor(x_input_scaled, dtype=torch.float32)

    with torch.no_grad():
        x_next_scaled = model(x_input_tensor).numpy()
    x_next = scaler_y.inverse_transform(x_next_scaled)
    xhat_nn[k+1] = x_next   
end_time_nn = time.perf_counter()
avg_time_nn = (end_time_nn - start_time_nn) / Nsim * 1e3
print(f"Avg. NN estimation time per iter: {avg_time_nn:.6f} ms")
# Compare MHE vs NN estimation
plt.figure(figsize=(10, Nx*2))
for i in range(Nx):
    plt.subplot(Nx,1,i+1)
    plt.plot(t, x_true[:,i], label='True')
    plt.plot(t, xhat[:,i], '--', label='MHE')
    plt.plot(t, xhat_nn[:,i], ':', label='NN Est.')
    plt.ylabel(f'$x_{i}$')
    plt.legend()
plt.xlabel("Time [s]")
plt.suptitle("State Estimation: MHE vs NN Approximation")
plt.tight_layout()
plt.show()


