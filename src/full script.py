# %% [markdown]
# ## Librairies 

# %%
import numpy as np
import mpctools as mpc
import mpctools.plots as mpcplots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import control as ctrl
import optuna
import time


# %% [markdown]
# ## State-space representation

# %%
# Longitudinal flight dynamics 
# Acont = [udot, wdot, qdot, thetadot, hdot]'
Acont = np.array([[-0.026, 0.074, -0.804, -9.809,0],
              [-0.242, -2.017, 73.297, -0.105, -0.001],
              [0.003, -0.135, -2.941, 0, 0],
              [0,       0,      1,    0,0],
              [-0.011, 1,        0 ,   -75, 0]])

# Bcont = [delta_th,delta_e]
Bcont = np.array([[4.594,0], [-0.0004, -13.735], [0.0002,-24.410], [0,0],[0,0]]) 

# Ccont = measured states theta and h
Ccont = np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])

Dcont = np.array([[0], [0]])

# %% [markdown]
# ## Controllability and observability

# %%
def pbh_controllability(A, B):
    n = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)

    for i in eigenvalues:
        rank_test = np.linalg.matrix_rank(np.hstack([A - i * np.eye(n), B]))
        if rank_test < n:
            print(f"System is NOT controllable. Uncontrollable mode at lambda = {i:.4f}")
            return False

    print("System is controllable.")
    return True

def pbh_observability(A, C):
    n = A.shape[0]
    eigenvalues = np.linalg.eigvals(A)

    for j in eigenvalues:
        rank_test = np.linalg.matrix_rank(np.vstack([j * np.eye(n) - A, C]))
        if rank_test < n:
            print(f"System is NOT observable. Unobservable mode at lambda = {j:.4f}")
            return False

    print("System is observable.")
    return True

# Check controllability and observability
pbh_controllability(Acont, Bcont)
pbh_observability(Acont, Ccont)

# %% [markdown]
# ## MPC design

# %%
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = 0.25
Nt = 20 # Pas de temps
(A, B) = mpc.util.c2d(Acont,Bcont,dt)

def ffunc(x,u):
    """Linear discrete-time model."""
    return mpc.mtimes(A, x) + mpc.mtimes(B, u)
f = mpc.getCasadiFunc(ffunc, [n, m], ["x", "u"], "f")

umax = [1, 0.262]  # [Throttle, Elevator]
umin = [0, -0.262]
dumax = [0.1, 0.524]  # Rate constraints
dumin = [-0.1, -0.524]

# State constraints
theta_max = 0.349  
theta_min = -0.349

x_min = [-np.inf, -np.inf, -np.inf, theta_min, -np.inf]  
x_max = [np.inf, np.inf, np.inf, theta_max, np.inf]

# Set constraints in MPC
lb = dict(u=umin, du=dumin, x=x_min)
ub = dict(u=umax, du=dumax, x=x_max)

# Define Q and R matrices.
Q = np.diag([1, 1/0.12, 1/0.012, 1/0.12, 1/0.0172])
R = np.diag([1/0.52, 1/0.17452])

def lfunc(x,u):
    """Quadratic stage cost."""
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)

l = mpc.getCasadiFunc(lfunc, [n,m], ["x","u"], "l")

# Initial condition and sizes.
x0 = np.array([15, 0, 0, 0, 100])  # [u, w, q, θ, h]
N = {"x" : n, "u" : m, "t" : Nt}


# %% [markdown]
# ## Solving MPC

# %%
# Solve MPC
solver = mpc.nmpc(f, l, N, x0, lb, ub,verbosity=0, isQP=True)
nsim = 40
t = np.arange(nsim + 1) * dt
xcl = np.zeros((n, nsim + 1))
xcl[:, 0] = x0
ucl = np.zeros((m, nsim))
ducl = np.zeros((m, nsim))
x4cl = np.zeros(nsim + 1)
for k in range(nsim):
    solver.fixvar("x", 0, x0)
    sol = mpc.callSolver(solver)
    print("Iteration %d Status: %s" % (k, sol["status"]))
    xcl[:, k] = x0
    x4cl[k] = x0[3]  
    ucl[:, k] = sol["u"][0, :]    
    if k > 0:
        ducl[:, k] = ucl[:, k] - ucl[:, k - 1]
    else:
        ducl[:, k] = ucl[:, k] - 0
    x0 = ffunc(x0, ucl[:, k])
xcl[:, nsim] = x0
x4cl[nsim] = x0[3]


# %% [markdown]
# ## Closed-loop response

# %%
# Convert x4 from radians to degrees
x4_degrees = np.degrees(xcl[3, :])  
theta_max_degrees = np.degrees(theta_max)  
theta_min_degrees = np.degrees(theta_min)

# Plot x4 
plt.figure(figsize=(10, 6))
plt.plot(t, x4_degrees, label='Pitch Angle θ (degrees)', color='b')
plt.axhline(theta_max_degrees, color='r', linestyle='--', label='θ max')
plt.axhline(theta_min_degrees, color='r', linestyle='--', label='θ min')
plt.xlabel('Time [s]')
plt.ylabel('Pitch Angle θ [deg]')
plt.title('Evolution of Pitch Angle θ over Time')
plt.legend()
plt.grid()
plt.show()

# Convert control inputs
umax_degrees = [umax[0], np.degrees(umax[1])]  
umin_degrees = [umin[0], np.degrees(umin[1])]
ucl_degrees = np.copy(ucl)
ucl_degrees[1, :] = np.degrees(ucl[1, :])  

# Plot throttle and elevator 
plt.figure(figsize=(10, 6))
plt.step(t[:-1], ucl_degrees[0, :], where='post', label='Throttle δ_t (unitary)', color='g')
plt.step(t[:-1], ucl_degrees[1, :], where='post', label='Elevator δ_e (degrees)', color='r')
plt.axhline(umax_degrees[1], color='r', linestyle='--', label='Elevator max')
plt.axhline(umin_degrees[1], color='r', linestyle='--', label='Elevator min')
plt.xlabel('Time [s]')
plt.ylabel('Control Inputs')
plt.title('Evolution of Control Inputs δ_t and δ_e over Time')
plt.legend()
plt.grid()
plt.show()

# Convert rate of change of control inputs
ducl_degrees = np.copy(ducl)
ducl_degrees[1, :] = np.degrees(ducl[1, :])  

dumax_degrees = [dumax[0], np.degrees(dumax[1])]
dumin_degrees = [dumin[0], np.degrees(dumin[1])]

# Plot control rate
plt.figure(figsize=(10, 6))
plt.step(t[:-1], ducl_degrees[0, :], where='post', label='ΔThrottle δ_t', color='g')
plt.step(t[:-1], ducl_degrees[1, :], where='post', label='ΔElevator δ_e (degrees/s)', color='r')
plt.axhline(dumax_degrees[1], color='r', linestyle='--', label='ΔElevator max')
plt.axhline(dumin_degrees[1], color='r', linestyle='--', label='ΔElevator min')
plt.xlabel('Time [s]')
plt.ylabel('Rate of Change of Control Inputs')
plt.title('Evolution of Control Input Rate (Δu) over Time')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 8))

state_labels = ["u (m/s)", "w (m/s)", "q (rad/s)", "θ (rad)", "h (m)"]
for i in range(n):  # n=5
    plt.subplot(n, 1, i+1)
    plt.plot(t, xcl[i, :], label=f"x[{i}] = {state_labels[i]}")
    plt.ylabel(state_labels[i])
    plt.grid(True)
    if i == 0:
        plt.title("Evolution of the 5 States Over Time")
    if i == n-1:
        plt.xlabel("Time [s]")
    plt.legend(loc="best")

plt.tight_layout()
plt.show()

# Compute the measured outputs y for each time step
ycl = np.zeros((Ccont.shape[0], nsim+1))  # shape: (2, nsim+1)

for k in range(nsim+1):
    # y = C*x
    ycl[:, k] = Ccont @ xcl[:, k]  # (2,)

# Plot the measured outputs
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, ycl[0, :], label="θ (rad)", color="b")
plt.ylabel("Pitch Angle [rad]")
plt.title("Measured Outputs Over Time")
plt.grid(True)
plt.legend(loc="best")

plt.subplot(2, 1, 2)
plt.plot(t, ycl[1, :], label="h (m)", color="g")
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.grid(True)
plt.legend(loc="best")

plt.tight_layout()
plt.show()



# %% [markdown]
# ## MHE 
# 

# %%
from scipy import linalg

np.random.seed(927) # Seed random number generator.

verb = 2
doPlots = True

# Problem parameters.
Delta = .25
Nt = 20
t = np.arange(Nt+1)*Delta

Nx = 5
Nu = 2
Ny = 2
Nw = Nx
Nv = Ny
Nc = 4

Gcont = Acont @ np.linalg.inv(linalg.expm(Acont * Delta) - np.eye(Acont.shape[0]))

def fcontinuous(x,u,w):
    return mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u) + mpc.mtimes(Gcont,w)
fcontinuous = mpc.getCasadiFunc(fcontinuous,[Nx,Nu,Nw],["x","u","w"],"f")

(A, B) = mpc.util.c2d(Acont, Bcont, Delta)
C = Ccont

def Fdiscrete(x,u,w):
    return mpc.mtimes(A, x) + mpc.mtimes(B, u) + w
Fdiscrete = mpc.getCasadiFunc(Fdiscrete,[Nx,Nu,Nw],["x","u","w"],"F")

def H(x): return mpc.mtimes(C, x)
H = mpc.getCasadiFunc(H,[Nx],["x"],"H")

# Noise covariances.
Q = 0.01 * np.diag([0.1, 0.25, 0.05, 0.01, 0.01])  
Qhalf = linalg.cholesky(Q,lower=True)
Qinv = linalg.inv(Q)

R = np.diag([.5,.25])
Rhalf = linalg.cholesky(R,lower=True)
Rinv = linalg.inv(R)

# First simulate the noisy system.
x0 = np.array([15, 0, 0, 0, 100]) 
x0hat = np.zeros(Nx)

omega = 2*np.pi/(Nt*Delta)
u = np.vstack((np.sin(omega*t),np.cos(omega*t))).T # Use periodic input.
u = u[:-1,:] # Get rid of final u (it isn't used).
w = Qhalf.dot(np.random.randn(Nw,Nt)).T
v = Rhalf.dot(np.random.randn(Nv,Nt+1)).T

x = np.zeros((Nt+1,Nx))
x[0,:] = x0
y = np.zeros((Nt+1,Ny))

for k in range(Nt+1):
    thisy = H(x[k,:])
    y[k,:] = np.squeeze(thisy) + v[k,:]
    if k < Nt:
        xnext = Fdiscrete(x[k,:],u[k,:], w[k,:])
        x[k+1,:] = np.squeeze(xnext)
    
# Plot simulation.
if doPlots:
    f = plt.figure()
    
    # State
    ax = f.add_subplot(2,1,1)
    for i in range(Nx):
        ax.plot(t,x[:,i],label="$x_{%d}$" % (i,))
    ax.set_ylabel("$x$")
    ax.legend()
    
    # Measurement
    ax = f.add_subplot(2,1,2)
    for i in range(Ny):
        ax.plot(t,y[:,i],label="$y_{%d}$" % (i,))
    ax.set_ylabel("$y$")
    ax.set_xlabel("Time")
    ax.legend()
    f.tight_layout(pad=.5)

# Now we're ready to try some state estimation. First define stage cost and
# prior.
def l(w,v):
    return mpc.mtimes(w.T,Qinv,w) + mpc.mtimes(v.T,Rinv,v)
l = mpc.getCasadiFunc(l,[Nw,Nv],["w","v"],"l")
def lx(x):
    return 10*mpc.mtimes(x.T,x)
lx = mpc.getCasadiFunc(lx,[Nx],["x"],"lx")

N = {"t" : Nt, "x" : Nx, "y" : Ny, "u" : Nu, "c" : Nc}

out = mpc.callSolver(mpc.nmhe(fcontinuous, H, u, y, l, N, lx, x0hat,
                              verbosity=5, Delta=Delta))

west = out["w"]

xest = out["x"]
xerr = xest - x

# Now we need to smush together all of the collocation points and actual
# points.
[T,X,Tc,Xc] = mpc.util.smushColloc(out["t"],out["x"],out["tc"],out["xc"])

# Plot estimation.
if doPlots:
    colors = ["red","blue","green","yellow","black"]    
    
    f = plt.figure()
    
    # State
    ax = f.add_subplot(2,1,1)
    for i in range(Nx):
        c = colors[i % len(colors)]
        ax.plot(T,X[:,i],label=r"$\hat{x}_{%d}$" % (i,),color=c)
        ax.plot(t,xest[:,i],"o",markeredgecolor=c,markerfacecolor=c,
                markersize=3.5)
        ax.plot(Tc,Xc[:,i],"o",markeredgecolor=c,markerfacecolor="none",
                markersize=3.5)
    ax.set_ylabel("$x$")
    ax.legend()
    
    # Measurement
    ax = f.add_subplot(2,1,2)
    for i in range(Nx):
        ax.plot(t,xerr[:,i],label=r"$\hat{x}_{%d} - x_{%d}$" % (i,i))
    ax.set_ylabel("Error")
    ax.set_xlabel("Time")
    ax.legend()
    f.tight_layout(pad=.5)
    mpc.plots.showandsave(f,"mpcexample.pdf")



