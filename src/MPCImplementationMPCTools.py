import numpy as np
import mpctools as mpc
import mpctools.plots as mpcplots
import matplotlib.pyplot as plt

# Define continuous time model.
Acont = np.array([[-1.2822, 0, 0.98, 0],
              [0, 0, 1, 0],
              [-5.4293, 0, -1.8366, 0],
              [-128.2, 128.2, 0, 0]])
Bcont = np.array([[-0.3], [0], [-17], [0]])
Ccont = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
Dcont = np.array([[0], [0]])

n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .25
Nt = 10
(A, B) = mpc.util.c2d(Acont,Bcont,dt)

def ffunc(x,u):
    return mpc.mtimes(A, x) + mpc.mtimes(B, u)
f = mpc.getCasadiFunc(ffunc, [n, m], ["x", "u"], "f")

# Bounds on u, du, and x.
umax = 0.262
umin = -0.262
dumax = 0.524
dumin = -0.524
x2max = 0.349
x2min = -0.349

x_min = [-np.inf, x2min, -np.inf, -np.inf]
x_max = [np.inf, x2max, np.inf, np.inf]

lb = dict(u=[umin], du=[dumin], x=x_min)
ub = dict(u=[umax], du=[dumax], x=x_max)

# Define Q and R matrices.
Q = np.diag([1, 10, 1, 1])
R = np.array([[100]])

def lfunc(x,u):
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)

l = mpc.getCasadiFunc(lfunc, [n,m], ["x","u"], "l")

# Initial condition and sizes.
x0 = np.array([0,0, 0, 100])
N = {"x" : n, "u" : m, "t" : Nt}

# Simulate
solver = mpc.nmpc(f, l, N, x0, lb, ub,verbosity=0, isQP=True)
nsim = 40
t = np.arange(nsim + 1) * dt
xcl = np.zeros((n, nsim + 1))
xcl[:, 0] = x0
ucl = np.zeros((m, nsim))
ducl = np.zeros((m, nsim))
x2cl = np.zeros(nsim + 1)
for k in range(nsim):
    solver.fixvar("x", 0, x0)
    sol = mpc.callSolver(solver)
    print("Iteration %d Status: %s" % (k, sol["status"]))
    xcl[:, k] = x0
    x2cl[k] = x0[1]  
    ucl[:, k] = sol["u"][0, :]    
    if k > 0:
        ducl[:, k] = ucl[:, k] - ucl[:, k - 1]
    else:
        ducl[:, k] = ucl[:, k] - 0
    x0 = ffunc(x0, ucl[:, k])
xcl[:, nsim] = x0
x2cl[nsim] = x0[1]

umax_degrees = np.degrees(umax)
umin_degrees = np.degrees(umin)
ucl_degrees = np.degrees(ucl)
plt.figure(figsize=(10, 6))
plt.plot(t, x2cl, label='x2 (second state variable)')
plt.xlabel('Time [s]')
plt.ylabel('x2')
plt.title('Evolution of x2 over Time')
plt.legend()
plt.grid()
plt.show()

# Plotting ucl 
plt.figure(figsize=(10, 6))
plt.step(t[:-1], ucl_degrees.flatten(), where='post', label='Control Input (u)')  # Using step plot
plt.axhline(umax_degrees, color='r', linestyle='--', label='u max')
plt.axhline(umin_degrees, color='r', linestyle='--', label='u min')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.title('Evolution of Control Input (u) over Time')
plt.legend()
plt.grid()
plt.show()

# Convert x2 from radians to degrees
x2_degrees = np.degrees(xcl[1, :])
x2max_degrees = np.degrees(x2max)
x2min_degrees = np.degrees(x2min)


# Plotting x2 and x4 
plt.figure(figsize=(10, 6))
plt.plot(t, x2_degrees, label='x2 (second state variable in degrees)')
plt.plot(t, xcl[3, :], label='x4 (fourth state variable)')
plt.axhline(x2max_degrees, color='r', linestyle='--')
plt.axhline(x2min_degrees, color='r', linestyle='--')
plt.xlabel('Time [s]')
plt.ylabel('State Variables')
plt.title('Evolution of x2 (in degrees) and x4 over Time')
plt.legend()
plt.grid()
plt.show()

ducl_degrees = np.degrees(ducl)
dumax_degrees = np.degrees(dumax)
dumin_degrees = np.degrees(dumin)

# Plotting ducl
plt.figure(figsize=(10, 6))
plt.step(t[:-1], ducl_degrees.flatten(), where='post', label='Ducl')  # Using step plot
plt.axhline(dumax_degrees, color='r', linestyle='--', label='dumax')
plt.axhline(dumin_degrees, color='r', linestyle='--', label='duumin')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.title('Evolution of du(t) over Time')
plt.legend()
plt.grid()
plt.show()



