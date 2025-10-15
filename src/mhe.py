import numpy as np
from scipy.linalg import expm, solve, inv, cholesky
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import time
from qpsolvers import solve_qp

def constraint_violated(xhat, lb, ub, tol=1e-8):
    return ~((xhat >= lb - tol) & (xhat <= ub + tol))

# System definition
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

# Discretize
Delta = 0.1
Nx, Nu, Ny = 5, 2, 3
A = expm(Acont * Delta)
B = np.linalg.solve(Acont, (A - np.eye(Nx))).dot(Bcont)

# Noise and horizon
Q = 0.01 * np.eye(Nx)
R = 0.1 * np.eye(Ny)
horizon = 25
Nsim = 50
t = np.arange(Nsim+1) * Delta

# State constraints
state_lb = np.array([7.0, -10.0, -1.0, -0.5, 0.0])
state_ub = np.array([100.0, 50.0, 1.0, 0.5, 200.0])

# MHE with constraints
class ConstrainedMHE:
    def __init__(self, A, B, C, Q, R, P0inv, lb, ub):
        self.A = A
        self.B = B
        self.C = C
        self.Qinv = inv(Q)
        self.Rinv = inv(R)
        self.Pinv = P0inv
        self.nx = A.shape[0]
        self.lb = lb
        self.ub = ub

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

        lb_all = np.tile(self.lb, M+1)
        ub_all = np.tile(self.ub, M+1)
        G = np.vstack([np.eye(Nvar), -np.eye(Nvar)])
        h = np.hstack([ub_all, -lb_all])

        x = solve_qp(H.astype(np.float64), -g.astype(np.float64),
                     G.astype(np.float64), h.reshape(-1, 1).astype(np.float64),
                     solver="mosek")
        if x is None:
            raise ValueError("QP solver failed.")
        return x[-n:]

# Simulate system
np.random.seed(1)
x_true = np.zeros((Nsim+1, Nx))
x_true[0] = np.zeros(Nx)
omega = 2 * np.pi / (Nsim * Delta)
u = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
y_meas = np.zeros((Nsim+1, Ny))
w = cholesky(Q, lower=True) @ np.random.randn(Nx, Nsim)
v = cholesky(R, lower=True) @ np.random.randn(Ny, Nsim+1)

for k in range(Nsim):
    x_true[k+1] = A @ x_true[k] + B @ u[k] + w[:, k]
    y_meas[k] = C @ x_true[k] + v[:, k]
y_meas[Nsim] = C @ x_true[Nsim] + v[:, Nsim]

# Run MHE
mhe = ConstrainedMHE(A, B, C, Q, R, inv(np.eye(Nx)), state_lb, state_ub)
xhat_mhe = np.zeros((Nsim+1, Nx))
xhat_mhe[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])  
start_time = time.perf_counter()

for k in range(Nsim):
    tmin = max(0, k - horizon)
    tmax = k + 1
    u_win = u[tmin:tmax-1] if k >= 1 else np.zeros((0, Nu))
    y_win = y_meas[tmin:tmax]
    x0bar = xhat_mhe[k].copy()
    xhat_mhe[k+1] = mhe.estimate(x0bar, u_win, y_win)

end_time = time.perf_counter()

# RMSE
def compute_rmse(xhat, xtrue):
    errors = np.sum((xhat - xtrue)**2, axis=1)
    rmse_total = np.sqrt(np.mean(errors))
    rmse_per_state = np.sqrt(np.mean((xhat - xtrue)**2, axis=0))
    return rmse_total, rmse_per_state

rmse_total, rmse_states = compute_rmse(xhat_mhe, x_true)
print(f"[MHE] Avg time per iter: {(end_time - start_time) / Nsim * 1e3:.3f} ms")
print(f"[MHE] RMSE (total): {rmse_total:.6f}")
for i, err in enumerate(rmse_states):
    print(f"    x_{i} RMSE: {err:.6f}")

# Plot estimates
plt.figure(figsize=(10, Nx*2))
for i in range(Nx):
    plt.subplot(Nx, 1, i+1)
    plt.plot(t, x_true[:, i], label='True')
    plt.plot(t, xhat_mhe[:, i], '--', label='MHE')
    plt.ylabel(f'$x_{i}$')
    plt.legend()
plt.xlabel("Time [s]")
plt.suptitle("Constrained MHE (no A-CL, no NN)")
plt.tight_layout()
plt.show()

# === MONTE CARLO EVALUATION FOR MHE ===
N_mc = 100
rng = np.random.default_rng(seed=42)

rmse_list = []
constraint_flags = []
timing_list = []

print("\nRunning Monte Carlo Evaluation (MHE)...")

for trial in range(N_mc):
    x0 = rng.uniform(low=[0, -1, -0.1, -0.1, 0], high=[20, 1, 0.1, 0.1, 50], size=Nx)
    x_true_mc = np.zeros((Nsim+1, Nx))
    y_meas_mc = np.zeros((Nsim+1, Ny))
    x_true_mc[0] = x0
    w_mc = cholesky(Q, lower=True) @ rng.standard_normal((Nx, Nsim))
    v_mc = cholesky(R, lower=True) @ rng.standard_normal((Ny, Nsim+1))
    u_mc = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]

    # Simulate true system
    for k in range(Nsim):
        x_true_mc[k+1] = A @ x_true_mc[k] + B @ u_mc[k] + w_mc[:, k]
        y_meas_mc[k] = C @ x_true_mc[k] + v_mc[:, k]
    y_meas_mc[Nsim] = C @ x_true_mc[Nsim] + v_mc[:, Nsim]

    # MHE estimation
    xhat_mc = np.zeros((Nsim+1, Nx))
    xhat_mc[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])  
    times = []

    for k in range(Nsim):
        start_k = time.perf_counter()

        tmin = max(0, k - horizon)
        u_win = u_mc[tmin:k] if k >= 1 else np.zeros((0, Nu))
        y_win = y_meas_mc[tmin:k+1]
        x0bar = xhat_mc[k].copy()
        xhat_mc[k+1] = mhe.estimate(x0bar, u_win, y_win)

        end_k = time.perf_counter()
        times.append((end_k - start_k) * 1e3)  # ms

    # Compute stats
    rmse_trial = compute_rmse(xhat_mc, x_true_mc)[0]
    constraints_ok = np.all((xhat_mc >= state_lb - 1e-8) & (xhat_mc <= state_ub + 1e-8))

    if not constraints_ok:
        print(f"\n Constraint violation at trial {trial}")
        print(f"Initial true state: {x0}")
        
        violations = constraint_violated(xhat_mc, state_lb, state_ub)

        if not np.all(~violations):
            print(f"\n⛔ Constraint violation at trial {trial}")
            print(f"Initial true state: {x0}")
            for i in range(Nsim+1):
                for j in range(Nx):
                    if violations[i, j]:
                        print(f"  At time {i*Delta:.2f}s: x_{j} = {xhat_mc[i,j]:.6f} violates [{state_lb[j]}, {state_ub[j]}]")


    # Store
    rmse_list.append(rmse_trial)
    constraint_flags.append(constraints_ok)
    timing_list.append(times)

    if (trial + 1) % 10 == 0:
        print(f"  Completed {trial+1}/{N_mc} runs")

# === Summary ===
rmse_mean = np.mean(rmse_list)
rmse_std = np.std(rmse_list)
constraint_count = sum(constraint_flags)
timing_array = np.array(timing_list)
timing_min = np.min(timing_array)
timing_max = np.max(timing_array)
timing_avg = np.mean(timing_array)

print("\n=== MHE Monte Carlo Summary ===")
print(f"RMSE: mean = {rmse_mean:.4f}, std = {rmse_std:.4f}")
print(f"Constraint-satisfying trajectories: {constraint_count}/{N_mc}")
print(f"Computation time per step [ms]: min = {timing_min:.3f}, max = {timing_max:.3f}, avg = {timing_avg:.3f}")



