import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt

# Define system matrices from the transfer function model.
A = np.array([[0.6]])
B = np.array([[0.05]])
n = A.shape[0]  # Number of states.
m = B.shape[1]  # Number of control elements

# Define sampling time and discretize the system.
dt = 1.0
(A, B) = mpc.util.c2d(A, B, dt)

def ffunc(x, u):
    """Linear discrete-time model."""
    return mpc.mtimes(A, x) + mpc.mtimes(B, u)

# Create the function with CasADi.
f = mpc.getCasadiFunc(ffunc, [n, m], ["x", "u"], "f")

# Define bounds and constraints.
umax = 0.5
umin = -0.5
x_max = [np.inf]
x_min = [-np.inf]

lb = dict(u=[umin], x=x_min)
ub = dict(u=[umax], x=x_max)

# Define Q and R matrices for cost.
Q = np.array([[1]])
R = np.array([[0.1]])

def lfunc(x, u):
    """Quadratic stage cost."""
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)

l = mpc.getCasadiFunc(lfunc, [n, m], ["x", "u"], "l")

# Set initial condition and configure MPC.
x0 = np.array([0])
N = {"x": n, "u": m, "t": 10}
solver = mpc.nmpc(f, l, N, x0, lb, ub, verbosity=0, isQP=True)

# Simulation parameters.
nsim = 50
xcl = np.zeros((n, nsim + 1))
ucl = np.zeros((m, nsim))

# Simulate the process.
for k in range(nsim):
    solver.fixvar("x", 0, x0)
    sol = mpc.callSolver(solver)
    xcl[:, k] = x0
    ucl[:, k] = sol["u"][0, :]
    x0 = ffunc(x0, ucl[:, k])
xcl[:, nsim] = x0

# Plot results.
plt.plot(np.arange(nsim + 1) * dt, xcl[0, :], label="Moisture Content")
plt.step(np.arange(nsim) * dt, ucl[0, :], where="post", label="Control Input")
plt.xlabel("Time [s]")
plt.ylabel("Process Variables")
plt.legend()
plt.grid()
plt.show()
