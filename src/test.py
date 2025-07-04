# Import the libraries 
import numpy as np
from numpy import random
from scipy import linalg
import mpctools as mpc
import matplotlib.pyplot as plt
from scipy.linalg import expm 
import time

# Aircraft longitudinal state-space system 
# System dynamics
Acont = np.array([
    [-0.026, 0.074, -0.804, -9.809, 0], # udot forward velocity
    [-0.242, -2.017, 73.297, -0.105, -0.001], # wdot vertical velocity
    [0.003, -0.135, -2.941, 0, 0], # qdot pitch rate
    [0, 0, 1, 0, 0], # thetadot pitch angle
    [-0.011, 1, 0, -75, 0]# hdot altitude
])

# Control input matrix :  throttle and elevator
Bcont = np.array([
    [4.594, 0],
    [-0.0004, -13.735],
    [0.0002, -24.410],
    [0, 0],
    [0, 0]
])
# Output matrix
Ccont = np.array([
    [1, 0, 0, 0, 0],    # xdot
    [0, 0, 0, 1, 0],    # theta
    [0, 0, 0, 0, 1]     # h
])


# MHE parameters
Delta = 0.1          # Sampling time
Nt = 25             # Horizon length
Nsim = 50            # Simulation length

Nx, Nu, Ny = 5, 2, 3 # System dimensions 
Nw, Nv, Nc = Nx, Ny, 4 # Nc -> # constraints Nv and Nv -> noise sizes

t = np.arange(Nsim + 1) * Delta # Time vector

# Gcont: Noise influence matrix (used for process noise in continuous-time)
Gcont = Acont @ np.linalg.inv(expm(Acont * Delta) - np.eye(Nx))

# Discretize continuous system for simulation
(A, B) = mpc.util.c2d(Acont, Bcont, Delta)

# Continuous-time dynamics function
def fcontinuous_func(x, u, w):
    return mpc.mtimes(Acont, x) + mpc.mtimes(Bcont, u) + mpc.mtimes(Gcont, w)
fcontinuous = mpc.getCasadiFunc(fcontinuous_func, [Nx, Nu, Nw], ["x", "u", "w"], "f")

# Discrete-time dynamics function
def Fdiscrete_func(x, u, w):
    return mpc.mtimes(A, x) + mpc.mtimes(B, u) + w
Fdiscrete = mpc.getCasadiFunc(Fdiscrete_func, [Nx, Nu, Nw], ["x", "u", "w"], "F")

# Measurement function
def H_func(x):
    return mpc.mtimes(Ccont, x)
H = mpc.getCasadiFunc(H_func, [Nx], ["x"], "H")

# Process noise covariance for model uncertainty
Q = 0.01 * np.eye(Nx)
Qhalf = linalg.cholesky(Q, lower=True)
Qinv = np.linalg.inv(Q)

# Measurement noise covariance for sensor uncertainty
R = 0.1 * np.eye(Ny)
Rhalf = linalg.cholesky(R, lower=True)
Rinv = np.linalg.inv(R)

# Simulate system with noise
random.seed(927)

x0 = np.array([0, 0, 0, 0.0, 0])   # initial true state
x0hat = np.zeros(Nx) # initial state estimate

# Generate control inputs
omega = 2 * np.pi / (Nsim * Delta)
u = np.vstack((np.sin(omega * t), np.cos(omega * t))).T[:-1]

# Generate process and measurement noise
w = Qhalf @ random.randn(Nw, Nsim) # process noise
w = w.T
v = Rhalf @ random.randn(Nv, Nsim + 1)  # measurement noise
v = v.T

# Allocate arrays for true states and measurements
x = np.zeros((Nsim + 1, Nx))
y = np.zeros((Nsim + 1, Ny))
x[0, :] = x0

# Simulate the true system -> discrete-time model + noise
for k in range(Nsim + 1):
    y[k] = np.squeeze(H(x[k])) + v[k] # measured output with noise
    if k < Nsim:
        x[k + 1] = np.squeeze(Fdiscrete(x[k], u[k], w[k])) # next state

# MHE 
# Cost function for noise residuals
def l_func(w, v):
    return mpc.mtimes(w.T, Qinv, w) + mpc.mtimes(v.T, Rinv, v)
l = mpc.getCasadiFunc(l_func, [Nw, Nv], ["w", "v"], "l")

# Cost function for initial state deviation
def lx_func(x):
    return 10 * mpc.mtimes(x.T, x)
lx = mpc.getCasadiFunc(lx_func, [Nx], ["x"], "lx")

xhat = np.zeros((Nsim + 1, Nx)) # estimated states
xhat[0] = x0hat
solver = None

"""
At each step we:

Gather data from the sliding window (past + present data).

Solve an optimization problem to estimate the states.

Store the most recent estimate for the next step.

"""

total_start_time = time.perf_counter()
iteration_times = []

for t_step in range(Nsim): 
    iter_start_time = time.perf_counter()

    # Determine the length of the current estimation window
    M = min(max(t_step,1), Nt) # number of steps inside the estimation window. Not a full window -> use all data available else fixed-length window over Nt. Prevent window 0 at start.
    N_window = {"t": M, "x": Nx, "y": Ny, "u": Nu, "c": Nc} # Define problem dimensions 
    tmin = max(0, t_step - Nt) # Window start time
    tmax = t_step + 1 # Window end time

    # Extract current window of inputs and outputs
    u_window = u[tmin:tmax-1] if t_step >= 1 else np.zeros((0, Nu)) # input history over window
    y_window = y[tmin:tmax] # output history over window

    # Initialize or update solver
    if solver is None or t_step < Nt: # Startup case or when window is still filling

        y_window = y[tmin:tmax]
        u_window = u[tmin:tmax-1] if t_step>=1 else np.zeros((0, Nu))

        M = y_window.shape[0] - 1   
        N_window = {"t": M, "x": Nx, "y": Ny, "u": Nu, "c": Nc}
        solver = mpc.nmhe(fcontinuous, H, u_window, y_window, l, N_window,
                          lx, xhat[t_step], verbosity=0,
                          Delta=Delta)
    else: # Reuse solver to avoid re-creating it and update u,y and x0bar   
        solver.par["u"] = list(u_window) 
        solver.par["y"] = list(y_window)
        solver.par["x0bar"] = xhat[t_step]

    iter_end_time = time.perf_counter()
    iteration_times.append(iter_end_time - iter_start_time)

    sol = mpc.callSolver(solver)
    if sol["status"] != "Solve_Succeeded":
        print(f"MHE failed at step {t_step}")
        break

    # Save the last state estimate of the window as current estimate
    xhat[t_step + 1] = sol["x"][-1]

total_end_time = time.perf_counter()

print(f"\nTotal MHE time: {(total_end_time - total_start_time)*1e3:.6f} ms")
print(f"Average time per iteration: {np.mean(iteration_times)*1e3:.6f} miliseconds")
print(f"Max time per iteration: {np.max(iteration_times)*1e3:.6f} ms")
print(f"Min time per iteration: {np.min(iteration_times)*1e3:.6f} ms")

# Plots
fig, axs = plt.subplots(Nx, 1, figsize=(8, Nx * 2))
for i in range(Nx):
    axs[i].plot(t, x[:, i], label=f"True $x_{i}$")
    axs[i].plot(t, xhat[:, i], '--', label=f"Estimated $x_{i}$")
    axs[i].set_ylabel(f"$x_{i}$")
    axs[i].legend()
plt.suptitle("MHE")
axs[-1].set_xlabel("Time [s]")  
plt.tight_layout()
plt.show()

from scipy.signal import place_poles

# Compute observer gain L
desired_poles = [0.6, 0.61, 0.62, 0.63, 0.64]  
L = place_poles(A.T, Ccont.T, desired_poles).gain_matrix.T

# Initialize observer estimate
xhat_acl = np.zeros((Nsim + 1, Nx))
xhat_acl[0] = np.zeros(Nx)  # same initial guess as MHE

observer_iteration_times = []
observer_start_time = time.perf_counter()

# Run observer simulation
for k in range(Nsim):
    iter_start_time = time.perf_counter()

    y_meas = y[k]  # noisy measurement
    xhat_acl[k+1] = A @ xhat_acl[k] + B @ u[k] + L @ (y_meas - Ccont @ xhat_acl[k])

    iter_end_time = time.perf_counter()
    observer_iteration_times.append(iter_end_time - iter_start_time)

observer_end_time = time.perf_counter()

print(f"\nA-CL Observer total time: {(observer_end_time - observer_start_time)*1e3:.6f} ms")
print(f"A-CL Observer average time per iteration: {np.mean(observer_iteration_times)*1e3:.8f} ms")
print(f"A-CL Observer max time per iteration: {np.max(observer_iteration_times)*1e3:.8f} ms")
print(f"A-CL Observer min time per iteration: {np.min(observer_iteration_times)*1e3:.8f} ms")

# Plot A-CL
fig, axs = plt.subplots(Nx, 1, figsize=(8, Nx * 2))
for i in range(Nx):
    axs[i].plot(t, x[:, i], label=f"True $x_{i}$")
    axs[i].plot(t, xhat_acl[:, i], '--', label=f"A-CL Observer $\hat{{x}}_{i}$")
    axs[i].set_ylabel(f"$x_{i}$")
    axs[i].legend()
plt.suptitle("State Estimation: A-CL")
axs[-1].set_xlabel("Time [s]")
plt.tight_layout()
plt.show()

# Plot comparing A-CL vs MHE computation time
plt.figure(figsize=(8,4))
plt.plot(range(len(iteration_times)), iteration_times, label="MHE Time per iteration")
plt.plot(range(len(observer_iteration_times)), observer_iteration_times, label="A-CL Time per iteration")
plt.xlabel("Simulation Step")
plt.ylabel("Time [s]")
plt.title("Computation Time per Iteration")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from scipy.signal import place_poles
desired_poles = [0.6, 0.61, 0.62, 0.63, 0.64]
L = place_poles(A.T, Ccont.T, desired_poles).gain_matrix.T

def lx_func_update(x, x0bar, Pinv):
    dx = x - x0bar
    return mpc.mtimes(dx.T, Pinv, dx)

lx = mpc.getCasadiFunc(
    lx_func_update,
    [Nx,      # x
     Nx,      # x0bar
     (Nx,Nx)  # Pinv
    ],
    ["x", "x0bar", "Pinv"],
    "lx"
)

P = np.eye(Nx)*1.0        
x0bar = x0hat.copy()      
Pinv = np.linalg.inv(P)

build_times = []
solve_times = []
total_times = []
total_start_time = time.perf_counter()

for t_step in range(Nsim):
    iter_total_start = time.perf_counter()

    iter_build_start = time.perf_counter()

    M = min(max(t_step, 1), Nt)
    N_window = {"t": M, "x": Nx, "y": Ny, "u": Nu, "c": Nc}
    tmin = max(0, t_step - Nt)
    tmax = t_step + 1

    u_window = u[tmin:tmax-1] if t_step >= 1 else np.zeros((0, Nu))
    y_window = y[tmin:tmax]

    if solver is None or t_step < Nt:
        M = y_window.shape[0] - 1
        N_window = {"t": M, "x": Nx, "y": Ny, "u": Nu, "c": Nc}
        solver = mpc.nmhe(fcontinuous, H, u_window, y_window, l, N_window,
                          lx, x0bar=x0hat, extrapar=dict(Pinv=np.linalg.inv(P)),
                          verbosity=0, Delta=Delta, inferargs=True)
    else:
        solver.par["u"] = list(u_window)
        solver.par["y"] = list(y_window)
        solver.par["x0bar"] = xhat[t_step]
        solver.par["Pinv"] = np.linalg.inv(P)

    iter_build_end = time.perf_counter()
    build_times.append(iter_build_end - iter_build_start)

    # SOLVE TIMER
    iter_solve_start = time.perf_counter()
    sol = mpc.callSolver(solver)
    iter_solve_end = time.perf_counter()
    solve_times.append(iter_solve_end - iter_solve_start)

    # Check solution
    if sol["status"] != "Solve_Succeeded":
        print(f"MHE failed at step {t_step}")
        break

    xhat[t_step + 1] = sol["x"][-1]

    # A-CL update for hybrid
    x0bar = A @ xhat[t_step + 1] + B @ u[t_step] + L @ (y[t_step + 1] - Ccont @ xhat[t_step + 1])
    P = (A - L @ Ccont) @ P @ (A - L @ Ccont).T + Q + L @ R @ L.T

    iter_total_end = time.perf_counter()
    total_times.append(iter_total_end - iter_total_start)
total_end_time = time.perf_counter()


print(f"Total time for {Nsim} iterations: {(total_end_time - total_start_time)*1e3:.6f} ms")
print(f"Avg. build time per iteration: {np.mean(build_times)*1e3:.6f} ms")
print(f"Avg. build time per iteration: {np.mean(build_times)*1e3:.6f} ms")
print(f"Avg. solve time per iteration: {np.mean(solve_times)*1e3:.6f} ms")
print(f"Avg. total iteration time     : {np.mean(total_times)*1e3:.6f} ms")

plt.figure(figsize=(10, 4))
plt.plot(build_times, label="Build time")
plt.plot(solve_times, label="Solve time")
plt.plot(total_times, label="Total time")
plt.xlabel("Simulation Step")
plt.ylabel("Time [s]")
plt.title("Timing Breakdown per Iteration")
plt.legend()    
plt.grid(True)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(Nx, 1, figsize=(8, Nx * 2))
for i in range(Nx):
    axs[i].plot(t, x[:, i], label=f"True $x_{i}$")
    axs[i].plot(t, xhat[:, i], '--', label=f"MHE + A-CL $\hat{{x}}_{i}$")
    axs[i].set_ylabel(f"$x_{i}$")
    axs[i].legend()
axs[-1].set_xlabel("Time [s]")
plt.suptitle("State Estimation: A-CL + MHE")
plt.tight_layout()
plt.show()

import numpy as np
from scipy.linalg import solve, inv

class LinearMHEPreEstimator:
    """
    Implements the analytic linear MHE with a pre-estimating observer (A-CL) based on Sui et al. (2010).
    """
    def __init__(self, A, B, C, Q, R, L, P, horizon):
        self.A = A
        self.B = B
        self.C = C
        self.Qinv = inv(Q)
        self.Rinv = inv(R)
        self.Pinv = inv(P)
        self.nx = A.shape[0]
        self.horizon = horizon

    def estimate(self, x0bar, u_seq, y_seq):
        """
        Estimate the state at the end of the window given:
        - x0bar: pre-estimated initial state (n,)
        - u_seq: inputs over the window, shape (M, nu)
        - y_seq: measurements over the window, shape (M+1, ny)
        Returns x_hat at time M.
        """
        M = y_seq.shape[0] - 1
        n = self.nx
        N_vars = (M + 1) * n

        # Build Hessian H and gradient g for the quadratic program
        H = np.zeros((N_vars, N_vars))
        g = np.zeros(N_vars)

        # Initial-state prior term
        H[0:n, 0:n] += self.Pinv
        g[0:n] += self.Pinv @ x0bar

        # Measurement terms
        for k in range(M + 1):
            idx = k * n
            H[idx:idx+n, idx:idx+n] += self.C.T @ self.Rinv @ self.C
            g[idx:idx+n] += self.C.T @ self.Rinv @ y_seq[k]

        # Dynamics terms (process noise elimination)
        for k in range(M):
            i = k * n
            j = (k + 1) * n
            A, B = self.A, self.B
            Qinv = self.Qinv

            # Hessian blocks
            H[i:i+n, i:i+n]     += A.T @ Qinv @ A
            H[i:i+n, j:j+n]     += -A.T @ Qinv
            H[j:j+n, i:i+n]     += -Qinv @ A
            H[j:j+n, j:j+n]     += Qinv

            # Gradient contributions from known input u[k]
            bu = B @ u_seq[k]
            g[i:i+n] += -A.T @ Qinv @ bu
            g[j:j+n] += Qinv @ bu

        # Solve the linear system H * X = g
        X = solve(H, g)
        # Return state estimate at the end of horizon
        return X[-n:]

# Example usage:
# Define system matrices (from your problem)
A = np.array([
    [-0.026, 0.074, -0.804, -9.809, 0],
    [-0.242, -2.017, 73.297, -0.105, -0.001],
    [0.003, -0.135, -2.941, 0, 0],
    [0, 0, 1, 0, 0],
    [-0.011, 1, 0, -75, 0]
])
B = np.array([[4.594, 0], [-0.0004, -13.735], [0.0002, -24.410], [0, 0], [0, 0]])
C = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
Q = 0.01 * np.eye(5)
R = 0.1 * np.eye(3)
# Observer gain L (computed separately)
from scipy.signal import place_poles
desired_poles = [0.6, 0.61, 0.62, 0.63, 0.64]
L = place_poles(A.T, C.T, desired_poles).gain_matrix.T
# Prior covariance P
P = np.eye(5)

# Construct the estimator
horizon = 25

mhe = LinearMHEPreEstimator(A, B, C, Q, R, L, P, horizon)

# Suppose at some time step we have:
x0bar = np.zeros(5)          # from pre-estimator (A-CL)
u_window = np.random.randn(horizon, 2)  # example inputs
y_window = np.random.randn(horizon+1, 3)  # example measurements

# Estimate the state at the end of the window
x_est = mhe.estimate(x0bar, u_window, y_window)
print("Estimated state at window end:", x_est)

