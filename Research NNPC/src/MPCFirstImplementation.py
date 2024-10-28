# Importing libraries
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import casadi as ca
import scipy.linalg
import scipy.signal
from scipy.integrate import solve_ivp

# State-space representation of the aircraft in the form xdot = Ax + Bu y = Cx
# x1 is the angle of attack in rad,
# x2 is the pitch angle in rad,
# x3 is the pitch rate in rad/sec,
# x4 is the altitude in meters,
# u is the elevator angle in rad

A = np.array([[-1.2822, 0, 0.98, 0],
              [0, 0, 1, 0],
              [-5.4293, 0, -1.8366, 0],
              [-128.2, 128.2, 0, 0]])
B = np.array([[-0.3], [0], [-17], [0]])
C = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])
D = np.array([[0], [0]])

# Define continuous system
sys_open_loop = scipy.signal.StateSpace(A, B, C, D)

# Eigenvalues of the open-loop system
eig_open_loop = np.linalg.eigvals(A)
print("Eigenvalues of the open-loop system:\n", eig_open_loop)

# Continuous-time step response
time, response = scipy.signal.step(sys_open_loop)

response_pitch = response[:, 0]  # Response of pitch angle
response_altitude = response[:, 1]  # Response of altitude

# Plot step response
plt.figure(figsize=(8, 5))
plt.plot(time, response_pitch, label='Pitch angle response')
plt.plot(time, response_altitude, label='Altitude response')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('Open-Loop Step Response')
plt.legend()
plt.grid()
plt.show()

# Constraints: |u| <= 0.262, |udot| <= 0.524, |x2| <= 0.349

# Discretize the dynamics
sampling_time = 0.25
discrete_sys_open_loop = scipy.signal.cont2discrete((A, B, C, D), sampling_time)
print("Discrete dynamics:\n", discrete_sys_open_loop)
A_d, B_d, C_d, D_d, dt = discrete_sys_open_loop

# Discrete-time step response
num_steps = 50  # Number of discrete time steps to simulate
u = np.ones((num_steps, 1))  # Step input
time_discrete = np.arange(0, num_steps * sampling_time, sampling_time)

# Simulate discrete-time step response using dlsim
_, y_discrete, _ = scipy.signal.dlsim((A_d, B_d, C_d, D_d, dt), u)

# Plot the discrete-time step response
plt.figure(figsize=(8, 5))
plt.step(time_discrete, y_discrete[:, 0], label='Pitch angle response', where='post')
plt.step(time_discrete, y_discrete[:, 1], label='Altitude response', where='post')
plt.xlabel('Time (seconds)')
plt.ylabel('Response')
plt.title('Discrete-Time Step Response')
plt.legend()
plt.grid()
plt.show()

# Setting up LQR control
Q = np.diag([1, 1, 1, 1])  # State cost matrix (adjust values to emphasize each state)
R = np.array([[100]])      # Control effort cost

# Solve the Riccati equation
P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)
print("Solution to Riccati equation:\n", P)

# Gain matrix K = (R + B_d'*P*B_d)^-1 * B_d' * P * A_d
K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d)
print("Gain matrix:\n", K)

# Closed-loop system with LQR control
A_d_cl = A_d - B_d @ K
B_d_cl = B_d
C_d_cl = C_d
D_d_cl = D_d
discrete_sys_closed_loop = (A_d_cl, B_d_cl, C_d_cl, D_d_cl, dt)

# Simulate the closed-loop response
x_dis = np.zeros((4, num_steps + 1))  # State history
u_dis = np.zeros(num_steps)           # Control history
udot_dis = np.zeros(num_steps)        # Control input derivative history
x_dis[:, 0] = np.array([0, 0, 0, 100])  # Initial state

for k in range(num_steps):
    u_dis[k] = float(-K @ x_dis[:, k])  # LQR control input as a scalar
    if k > 0:
        udot_dis[k] = (u_dis[k] - u_dis[k - 1]) / sampling_time
    x_dis[:, k + 1] = A_d @ x_dis[:, k] + B_d.flatten() * u_dis[k]  # State update

# Plot x2, u, and u_dot in discrete time
time_dis = np.arange(num_steps + 1) * sampling_time

plt.figure(figsize=(10, 7))

# x2 (Pitch angle) response
plt.subplot(3, 1, 1)
plt.step(time_dis, x_dis[1, :], where='post')
plt.xlabel('Time (s)')
plt.ylabel('x2')
plt.title('Discrete-time Response of x2 (Pitch angle)')

# Control input (u)
plt.subplot(3, 1, 2)
plt.step(time_dis[:-1], u_dis, where='post')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('Discrete-time Control Input (u)')

# Derivative of control input (u_dot)
plt.subplot(3, 1, 3)
plt.step(time_dis[:-1], udot_dis, where='post')
plt.xlabel('Time (s)')
plt.ylabel('Derivative of Control Input (u_dot)')
plt.title('Discrete-time Derivative of Control Input (u_dot)')

plt.tight_layout()
plt.show()

# MPC parameters
N = 10  # Horizon length
nx = A_d.shape[0]  # Number of states
nu = B_d.shape[1]  # Number of inputs
Q = np.eye(nx)  # State cost matrix
R = 100 * np.eye(nu)  # Control cost matrix

# Constraints
u_max = 0.262
u_min = -0.262
du_max = 0.524 
du_min = -0.524
x2_max = 0.349
x2_min = -0.349

# Initial conditions
x0 = np.array([0, 0, 0, 100])  # Initial state
num_steps = 50  # Total simulation steps

# Initialize storage for simulation results
x_history = np.zeros((nx, num_steps + 1))
u_history = np.zeros(num_steps)
x_history[:, 0] = x0

# Create CasADi variables
x = ca.SX.sym('x', nx)
u = ca.SX.sym('u', nu)

# Create a function for system dynamics
f = ca.Function('f', [x, u], [A_d @ x + B_d.flatten() * u])

# MPC setup
opt_vars = []
opt_params = []
constraints = []
cost_function = 0

# Decision variables
X = ca.SX.sym('X', nx, N + 1)
U = ca.SX.sym('U', nu, N)

# Parameters (initial state)
P = ca.SX.sym('P', nx)

# Initialize constraints and cost
g = []
lbg = []
ubg = []

# Build the optimization problem
for k in range(N):
    if k == 0:
        x_prev = X[:, k]
        u_prev = U[:, k]
    else:
        x_prev = X[:, k]
        u_prev = U[:, k-1]
    # Dynamics constraint
    x_next = f(x_prev, U[:, k])
    g.append(X[:, k+1] - x_next)
    lbg += [0]*nx
    ubg += [0]*nx
    # Cost function
    cost_function += ca.mtimes([(X[:, k] - np.zeros(nx)).T, Q, (X[:, k] - np.zeros(nx))]) \
                     + ca.mtimes([U[:, k].T, R, U[:, k]])
    # Input constraints
    g.append(U[:, k])
    lbg += [u_min]
    ubg += [u_max]
    # Rate of change constraint
    if k > 0:
        du = U[:, k] - U[:, k-1]
        g.append(du)
        lbg += [du_min]
        ubg += [du_max]
    # State constraints on x2
    g.append(X[1, k])  # x2 is the second state
    lbg += [x2_min]
    ubg += [x2_max]

# Terminal cost
cost_function += ca.mtimes([(X[:, N] - np.zeros(nx)).T, Q, (X[:, N] - np.zeros(nx))])

# Initial condition constraint
g.append(X[:, 0] - P)
lbg += [0]*nx
ubg += [0]*nx

# Decision variables vector
opt_variables = ca.vertcat(
    ca.reshape(X, nx*(N+1), 1),
    ca.reshape(U, nu*N, 1)
)

# Optimization problem setup
nlp_problem = {
    'f': cost_function,
    'x': opt_variables,
    'g': ca.vertcat(*g),
    'p': P
}

# Solver options
opts = {
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.max_iter': 1000,
    'ipopt.tol': 1e-8
}

# Create solver instance
solver = ca.nlpsol('solver', 'ipopt', nlp_problem, opts)

# Simulation loop
for t in range(num_steps):
    # Initial guess
    x0_guess = np.tile(x_history[:, t], (N+1, 1)).flatten()
    u0_guess = np.zeros((nu*N, 1))
    init_guess = np.concatenate([x0_guess, u0_guess.flatten()])
    # Parameter (initial state)
    p = x_history[:, t]
    # Solve the optimization problem
    sol = solver(
        x0=init_guess,
        lbx=-ca.inf,
        ubx=ca.inf,
        lbg=lbg,
        ubg=ubg,
        p=p
    )
    # Extract control input
    opt_sol = sol['x'].full().flatten()
    X_opt = opt_sol[:nx*(N+1)].reshape((nx, N+1))
    U_opt = opt_sol[nx*(N+1):].reshape((nu, N))
    u_applied = U_opt[:, 0]
    # Apply control input and update state
    x_next = A_d @ x_history[:, t] + B_d.flatten() * u_applied
    x_history[:, t+1] = x_next
    u_history[t] = u_applied
    print(f"Time step {t}, Control Input: {u_applied}, State: {x_next}")

# Plotting the results
time = np.arange(num_steps + 1) * sampling_time

plt.figure(figsize=(12, 8))

# Plot x2 (Pitch angle)
plt.subplot(3, 1, 1)
plt.step(time, x_history[1, :], where='post')
plt.axhline(x2_max, color='r', linestyle='--')
plt.axhline(x2_min, color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('x2 (Pitch angle)')
plt.title('MPC-Controlled Pitch Angle (x2)')
plt.grid()

# Plot control input u
plt.subplot(3, 1, 2)
plt.step(time[:-1], u_history, where='post')
plt.axhline(u_max, color='r', linestyle='--')
plt.axhline(u_min, color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Control Input (u)')
plt.title('MPC-Controlled Elevator Angle (u)')
plt.grid()

# Plot derivative of control input u_dot
u_dot = np.diff(u_history) / sampling_time
plt.subplot(3, 1, 3)
plt.step(time[1:-1], u_dot, where='post')
plt.axhline(du_max / sampling_time, color='r', linestyle='--')
plt.axhline(du_min / sampling_time, color='r', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Derivative of Control Input (u_dot)')
plt.title('MPC-Controlled Derivative of Elevator Angle (u_dot)')
plt.grid()

plt.tight_layout()
plt.show()

