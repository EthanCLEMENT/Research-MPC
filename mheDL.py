import numpy as np
from numpy import random
from scipy.linalg import expm, inv, cholesky, solve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from qpsolvers import solve_qp

# check constraints on states and input
def check_state_bounds(x, x_lb, x_ub, tol=1e-8):
    violations = (x < x_lb - tol) | (x > x_ub + tol)
    return not np.any(violations), violations

def check_input_bounds(u, u_lb, u_ub, tol=1e-8):
    violations = (u < u_lb - tol) | (u > u_ub + tol)
    return not np.any(violations), violations

# sys
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

# Discretize the continuous system to a discrete-time model
A = expm(Acont*Delta) 
B = np.linalg.solve(Acont, (A - np.eye(Nx))).dot(Bcont)

Q = 0.01 * np.eye(Nx)
R = 0.1  * np.eye(Ny)

# MHE no ACL
"""
min || x0 - x0
"""
class LinearMHE:
    def __init__(self, A, B, C, Q, R, Pinv, x_lb, x_ub):
        self.A = A; self.B = B; self.C = C
        self.Qinv = inv(Q); self.Rinv = inv(R); self.Pinv = Pinv
        self.nx = A.shape[0]
        self.lb = x_lb; self.ub = x_ub

    def estimate(self, x0bar, u_seq, y_seq):
        M = y_seq.shape[0] - 1
        n = self.nx
        Nvar = (M+1)*n
        H = np.zeros((Nvar, Nvar))
        g = np.zeros(Nvar)

        # Prior
        H[0:n,0:n] += self.Pinv
        g[0:n] += self.Pinv @ x0bar

        # Measurement
        for k in range(M+1):
            i = k*n
            H[i:i+n, i:i+n] += self.C.T @ self.Rinv @ self.C
            g[i:i+n]       += self.C.T @ self.Rinv @ y_seq[k]

        # Dynamics
        for k in range(M):
            i = k*n; j = (k+1)*n
            H[i:i+n, i:i+n] += self.A.T @ self.Qinv @ self.A
            H[i:i+n, j:j+n] += -self.A.T @ self.Qinv
            H[j:j+n, i:i+n] += -self.Qinv @ self.A
            H[j:j+n, j:j+n] += self.Qinv
            bu = self.B @ u_seq[k]
            g[i:i+n] += -self.A.T @ self.Qinv @ bu
            g[j:j+n] +=  self.Qinv @ bu

        # Box constraints
        lb_all = np.tile(self.lb, M+1)
        ub_all = np.tile(self.ub, M+1)
        G = np.vstack([np.eye(Nvar), -np.eye(Nvar)])
        h = np.hstack([ub_all, -lb_all])

        H = 0.5 * (H + H.T)
        x = solve_qp(H.astype(np.float64), -g.astype(np.float64),
                     G.astype(np.float64), h.astype(np.float64), solver="mosek")
        if x is None:
            raise ValueError("QP failed.")
        return x[-n:]

state_lb = np.array([7.0, -10.0, -1.0, -0.5, 0.0])
state_ub = np.array([100.0, 10.0, 1.0, 0.5, 200.0])
control_lb = np.array([-1.0, -1.0])
control_ub = np.array([1.0, 1.0])

Pinv0 = inv(np.eye(Nx))
mhe = LinearMHE(A, B, C, Q, R, Pinv0, state_lb, state_ub)

# dataset for NN 
random.seed(42)
omega = 2*np.pi/(Nsim*Delta)
num_simulations = 500
X_data, Y_data = [], []

for sim in range(num_simulations):
    x_sim = np.zeros((Nsim+1, Nx))
    u_sim = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
    y_sim = np.zeros((Nsim+1, Ny))

    w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
    v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)

    for k in range(Nsim):
        x_sim[k+1] = A @ x_sim[k] + B @ u_sim[k] + w[:,k]
        y_sim[k] = C @ x_sim[k] + v[:,k]
    y_sim[Nsim] = C @ x_sim[Nsim] + v[:,Nsim]

    xhat_sim = np.zeros((Nsim+1, Nx))
    xhat_sim[0] = np.array([20, 1, 0.1, 0.2, 100])
    x0bar = xhat_sim[0].copy()

    for k in range(Nsim):
        tmin = max(0, k-25)
        tmax = k+1
        u_win = u_sim[tmin:tmax-1] if k>=1 else np.zeros((0, Nu))
        y_win = y_sim[tmin:tmax]
        x0bar = xhat_sim[k].copy()
        
        xhat_sim[k+1] = mhe.estimate(x0bar, u_win, y_win)

        # Input for NN: [xhat_k, u_k, y_{k+1}]
        X_data.append(np.hstack([xhat_sim[k], u_sim[k], y_sim[k+1]]))
        # Label for NN: [xhat_{k+1}, u_k]
        Y_data.append(np.hstack([xhat_sim[k+1], u_sim[k]]))

X_data = np.array(X_data)
Y_data = np.array(Y_data)

# Train NN
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

class MHE_Net_XU(nn.Module):
    """
    Goal is to predict the next state estimate x_k+1 + constraint satisfying input
    """
    def __init__(self, in_dim, h, nx, nu, x_lb, x_ub, u_lb, u_ub):
        # nu = input dimension, nx = state dimension, 
        super().__init__()
        self.x_lb = nn.Parameter(torch.tensor(x_lb, dtype=torch.float32), requires_grad=False) 
        self.x_ub = nn.Parameter(torch.tensor(x_ub, dtype=torch.float32), requires_grad=False)
        self.u_lb = nn.Parameter(torch.tensor(u_lb, dtype=torch.float32), requires_grad=False)
        self.u_ub = nn.Parameter(torch.tensor(u_ub, dtype=torch.float32), requires_grad=False)
        self.backbone = nn.Sequential(nn.Linear(in_dim,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU()) #Linear -> relu -> linear -> relu
        self.head_x = nn.Linear(h, nx)
        self.head_u = nn.Linear(h, nu)
    def forward(self,z):
        # each batch concatenated as [xhat_k+1 u_k]
        h = self.backbone(z)
        rx = torch.tanh(self.head_x(h)) # raw predictions from head
        x = 0.5*(rx+1)*(self.x_ub-self.x_lb)+self.x_lb # map from (-1, 1) to physical range 
        ru = self.head_u(h) # raw predictions from head
        u = (self.u_ub-self.u_lb)/torch.pi*torch.atan(ru)+0.5*(self.u_ub+self.u_lb)
        return torch.cat([x,u],dim=1)

model = MHE_Net_XU(X_train.shape[1],64,Nx,Nu,state_lb,state_ub,control_lb,control_ub)
opt = optim.Adam(model.parameters(),lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop, xb = nx + nu + ny (dim 10) input batch, yb = nx + nu (dim 7) output batch
for epoch in range(100):
    model.train(); tot=0
    for xb,yb in train_loader:
        opt.zero_grad()
        pred=model(xb) # Pass the input batch through the network to get predictions
        loss=loss_fn(pred,yb) # Compare predictions to the ground truth targets in yb using MSE
        loss.backward(); opt.step() # Backpropagate the error and update weights
        tot+=loss.item()
    if (epoch+1)%10==0:
        print(f"Epoch {epoch+1}, Loss {tot/len(train_loader):.4f}")

# Monte Carlo evaluation
def compute_rmse(xhat,xtrue):
    return np.sqrt(np.mean(np.sum((xhat-xtrue)**2,axis=1)))

N_mc=100
rng=np.random.default_rng(0)
results={"MHE":{"rmse":[], "state_ok":[], "input_ok":[], "times":[]},
         "NN":{"rmse":[], "state_ok":[], "input_ok":[], "times":[]}}

model.eval()

for trial in range(N_mc):
    x0=rng.uniform([5,-1,-0.1,-0.1,10],[20,1,0.1,0.1,50]) #random initial state
    u_mc=np.vstack([np.sin(omega*t),np.cos(omega*t)]).T[:-1] # control input, T[:-1] drops last sample
    w_mc=cholesky(Q,lower=True)@rng.standard_normal((Nx,Nsim)) #process noise / step
    v_mc=cholesky(R,lower=True)@rng.standard_normal((Ny,Nsim+1)) # measurement noise / step
    x_true_mc=np.zeros((Nsim+1,Nx))
    y_meas_mc=np.zeros((Nsim+1,Ny))
    x_true_mc[0]=x0
    for k in range(Nsim):
        x_true_mc[k+1]=A@x_true_mc[k]+B@u_mc[k]+w_mc[:,k]
        y_meas_mc[k]=C@x_true_mc[k]+v_mc[:,k]
    y_meas_mc[Nsim]=C@x_true_mc[Nsim]+v_mc[:,Nsim]

    # MHE 
    xhat_mhe=np.zeros((Nsim+1,Nx))
    xhat_mhe[0]=np.array([20,1,0.1,0.2,100])
    x0bar=xhat_mhe[0].copy()
    times_mhe=[]
    for k in range(Nsim):
        tmin=max(0,k-25) # horizon sliding over 25 samples
        u_win=u_mc[tmin:k] if k>=1 else np.zeros((0,Nu))
        y_win=y_meas_mc[tmin:k+1] # measured outputs over the same window
        t0=time.perf_counter()
        x0bar = xhat_mhe[k].copy()
        xhat_mhe[k+1] = mhe.estimate(x0bar, u_win, y_win)
        times_mhe.append((time.perf_counter()-t0)*1e3)
    rmse_mhe=compute_rmse(xhat_mhe,x_true_mc)
    ok_s,_=check_state_bounds(xhat_mhe,state_lb,state_ub)
    ok_u,_=check_input_bounds(u_mc,control_lb,control_ub)
    results["MHE"]["rmse"].append(rmse_mhe)
    results["MHE"]["state_ok"].append(ok_s)
    results["MHE"]["input_ok"].append(ok_u)
    results["MHE"]["times"].append(times_mhe)

    viol_trials = [i for i,ok in enumerate(results["MHE"]["state_ok"]) if not ok]
    print("MHE state violation trials:", viol_trials)

    # NN 
    xhat_nn=np.zeros((Nsim+1,Nx)); uhat_nn=np.zeros((Nsim,Nu))
    xhat_nn[0]=np.array([20,1,0.1,0.2,100])
    times_nn=[]
    for k in range(Nsim):
        inp=np.hstack([xhat_nn[k],u_mc[k],y_meas_mc[k+1]])
        inp_t=torch.tensor(scaler_X.transform(inp.reshape(1,-1)),dtype=torch.float32)
        t0=time.perf_counter()
        with torch.no_grad():
            pred=model(inp_t).numpy()[0]
        times_nn.append((time.perf_counter()-t0)*1e3)
        xhat_nn[k+1]=pred[:Nx]; uhat_nn[k]=pred[Nx:]
    rmse_nn=compute_rmse(xhat_nn,x_true_mc)
    ok_s,_=check_state_bounds(xhat_nn,state_lb,state_ub)
    ok_u,_=check_input_bounds(uhat_nn,control_lb,control_ub)
    results["NN"]["rmse"].append(rmse_nn)
    results["NN"]["state_ok"].append(ok_s)
    results["NN"]["input_ok"].append(ok_u)
    results["NN"]["times"].append(times_nn)

    if (trial+1)%10==0: print(f"Completed {trial+1}/{N_mc}")

def summarize(name):
    rmse=np.mean(results[name]["rmse"])
    so=np.sum(results[name]["state_ok"]); io=np.sum(results[name]["input_ok"])
    time_arr=np.array(results[name]["times"])
    print(f"\n--- {name} ---")
    print(f"Avg RMSE {rmse:.4f}")
    print(f"State constraints ok {so}/{N_mc}")
    print(f"Input constraints ok {io}/{N_mc}")
    print(f"Computation time per step [ms]: "
          f"min={time_arr.min():.3f}, max={time_arr.max():.3f}, avg={time_arr.mean():.3f}")

summarize("MHE")
summarize("NN")

# Plots
# Histogram of RMSE
plt.figure()
plt.hist(results["MHE"]["rmse"], bins=15, alpha=0.6, label="MHE")
plt.hist(results["NN"]["rmse"], bins=15, alpha=0.6, label="NN")
plt.xlabel("RMSE")
plt.ylabel("Frequency")
plt.legend()
plt.title("RMSE Distribution")
plt.show()

# Example trajectory plot 
trial_id=0
plt.figure(figsize=(10,6))
x_true=np.zeros((Nsim+1,Nx))
xhat_nn=np.zeros((Nsim+1,Nx))

rng=np.random.default_rng(0)
x0=rng.uniform([5,-1,-0.1,-0.1,10],[20,1,0.1,0.1,50])
u_mc=np.vstack([np.sin(omega*t),np.cos(omega*t)]).T[:-1]
w_mc=cholesky(Q,lower=True)@rng.standard_normal((Nx,Nsim))
v_mc=cholesky(R,lower=True)@rng.standard_normal((Ny,Nsim+1))
x_true[0]=x0
y_meas=np.zeros((Nsim+1,Ny))
for k in range(Nsim):
    x_true[k+1]=A@x_true[k]+B@u_mc[k]+w_mc[:,k]
    y_meas[k]=C@x_true[k]+v_mc[:,k]
y_meas[Nsim]=C@x_true[Nsim]+v_mc[:,Nsim]
xhat_nn[0]=np.array([20,1,0.1,0.2,100])
for k in range(Nsim):
    inp=np.hstack([xhat_nn[k],u_mc[k],y_meas[k+1]])
    inp_t=torch.tensor(scaler_X.transform(inp.reshape(1,-1)),dtype=torch.float32)
    with torch.no_grad():
        pred=model(inp_t).numpy()[0]
    xhat_nn[k+1]=pred[:Nx]

for i in range(Nx):
    plt.plot(t, x_true[:,i], label=f"x{i}_true")
    plt.plot(t, xhat_nn[:,i], '--', label=f"x{i}_NN")
    plt.fill_between(t, state_lb[i], state_ub[i], color="gray", alpha=0.1)
plt.xlabel("Time [s]")
plt.ylabel("State values")
plt.title("True vs NN estimates (trial 0)")
plt.legend()
plt.show()

# Plot computation time distribution
plt.figure()
mhe_times=np.array(results["MHE"]["times"]).flatten()
nn_times=np.array(results["NN"]["times"]).flatten()
plt.hist(mhe_times, bins=20, alpha=0.6, label="MHE")
plt.hist(nn_times, bins=20, alpha=0.6, label="NN")
plt.xlabel("Per-step time [ms]")
plt.ylabel("Frequency")
plt.title("Computation time distribution")
plt.legend()
plt.show()