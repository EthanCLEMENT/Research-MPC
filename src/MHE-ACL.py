import numpy as np
from numpy import random
from scipy.linalg import expm, solve, inv, cholesky
from scipy.signal import place_poles
import matplotlib.pyplot as plt
import time
from qpsolvers import solve_qp


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
C = np.array([
    [1, 0, 0, 0, 0],    # xdot
    [0, 0, 0, 1, 0],    # theta
    [0, 0, 0, 0, 1]     # h
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
# Builds the Hessian H and gradient g over a sliding window and solves in closed-form
class LinearMHEPreEstimator:
    def __init__(self, A, B, C, Q, R, Pinv):
        self.A = A; self.B = B; self.C = C
        self.Qinv = inv(Q); self.Rinv = inv(R); self.Pinv = Pinv
        self.nx = A.shape[0]

    def estimate(self, x0bar, u_seq, y_seq):
        # M = window length-1, n = state dimension
        M = y_seq.shape[0] - 1
        n = self.nx
        Nvar = (M+1)*n
        H = np.zeros((Nvar, Nvar))
        g = np.zeros(Nvar)

        # Initial‐state prior
        H[0:n,0:n] += self.Pinv
        g[0:n]     += self.Pinv @ x0bar

        # Measurement terms
        # For each time k in horizon, add C^T R^{-1} C block and C^T R^{-1} y_k
        for k in range(M+1):
            i = k*n
            H[i:i+n, i:i+n] += self.C.T @ self.Rinv @ self.C
            g[i:i+n]       += self.C.T @ self.Rinv @ y_seq[k]
        # Dynamics terms
        # For k=0 to M-1, add A, Q^{-1} blocks linking x_k and x_{k+1}
        for k in range(M):
            i = k*n; j = (k+1)*n
            H[i:i+n, i:i+n] += self.A.T @ self.Qinv @ self.A
            H[i:i+n, j:j+n] += -self.A.T @ self.Qinv
            H[j:j+n, i:i+n] += -self.Qinv @ self.A
            H[j:j+n, j:j+n] += self.Qinv
            # Include control input effect
            bu = self.B @ u_seq[k]
            g[i:i+n] += -self.A.T @ self.Qinv @ bu
            g[j:j+n] +=  self.Qinv @ bu
            
    

        from qpsolvers import solve_qp
        # Constraints [u, w, q, theta, h]
        state_lb = np.array([7.0, -10.0, -1.0, -0.5, 0.0])
        state_ub = np.array([100.0, 10.0, 1.0, 0.5, 200.0])

        lb = np.tile(state_lb, M+1)
        ub = np.tile(state_ub, M+1)

        lb = np.tile(state_lb, M+1)
        ub = np.tile(state_ub, M+1)

        G = np.vstack([
            np.eye(Nvar),      
            -np.eye(Nvar)     
        ])
        h = np.hstack([ub, -lb])


        from cvxopt import matrix as cvxmat

        # Solve QP: minimize 0.5 x^T H x - g^T x  s.t. G x <= h
        # DO NOT use cvxmat() with qpsolvers.solve_qp
        # Just reshape h properly and ensure float64 type
        x = solve_qp(
        H.astype(np.float64),
        -g.astype(np.float64),
        G.astype(np.float64),
        h.reshape(-1, 1).astype(np.float64),  # make sure h is (m,1)
        solver="mosek"
        )   



        if x is None:
            raise ValueError("QP solver failed to find a solution.")
        
        return x[-n:]  # return last state in window


# Precompute inverses
Qinv = inv(Q)
Rinv = inv(R)
P0 = np.eye(Nx)     # initial prior cov
Pinv0 = inv(P0)

# Build estimator 
horizon = 25
mhe = LinearMHEPreEstimator(A, B, C, Q, R, Pinv0)

# --- Simulation of true system ---
random.seed(927)
x_true = np.zeros((Nsim+1, Nx))
y_meas = np.zeros((Nsim+1, Ny))
x_true[0] = np.zeros(Nx)
omega = 2*np.pi/(Nsim*Delta)
u = np.vstack([np.sin(omega*t), np.cos(omega*t)]).T[:-1]
# Generate zero-mean Gaussian noise via Cholesky
w = cholesky(Q, lower=True) @ random.randn(Nx, Nsim)
v = cholesky(R, lower=True) @ random.randn(Ny, Nsim+1)
for k in range(Nsim+1):
    # measurements y_k = C x_k + v_k
    y_meas[k] = C @ x_true[k] + v[:,k]
    if k<Nsim:
        # true state propagation x_{k+1} = A x_k + B u_k + w_k
        x_true[k+1] = A @ x_true[k] + B @ u[k] + w[:,k]

# --- Estimation loop ---
xhat = np.zeros((Nsim+1, Nx))
xhat[0] = np.array([20.0, 1.0, 0.1, 0.2, 100.0])  
x0bar = xhat[0].copy()
P = P0.copy()

times = []
total_start_time = time.perf_counter()

for k in range(Nsim):
    t0 = time.perf_counter()

    # Build window
    M = min(max(k,1), horizon)
    # Determine window indices [tmin, tmax]
    tmin = max(0, k-horizon)
    tmax = k+1
    u_win = u[tmin:tmax-1] if k>=1 else np.zeros((0, Nu))
    y_win = y_meas[tmin:tmax]

    # Pre‐estimator update (A-CL)
    if k>0:
        # x0bar = A xhat[k] + B u_{k-1} + L (y_k - C xhat[k])
        x0bar = A @ xhat[k] + B @ u[k-1] + L @ (y_meas[k] - C @ xhat[k])
        # covariance update P = (A-LC) P (A-LC)^T + Q + L R L^T
        P = (A - L@C)@P@(A - L@C).T + Q + L@R@L.T
        mhe.Pinv = inv(P)

    # MHE analytic solve
    xhat[k+1] = mhe.estimate(x0bar, u_win, y_win)

    times.append(time.perf_counter() - t0)
total_end_time = time.perf_counter()

# Compute squared error at each time step
squared_errors = np.sum((xhat - x_true)**2, axis=1)

# Mean squared error over all time steps
mse = np.mean(squared_errors)

# Root mean squared error
rmse = np.sqrt(mse)

print(f"RMSE over all states and time: {rmse:.6f}")
rmse_per_state = np.sqrt(np.mean((xhat - x_true)**2, axis=0))
for i in range(Nx):
    print(f"RMSE for state x_{i}: {rmse_per_state[i]:.6f}")



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
