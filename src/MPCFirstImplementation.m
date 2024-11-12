% Define system matrices
A = [-1.2822 0 0.98 0; 0 0 1 0; -5.4293 0 -1.8366 0; -128.2 128.2 0 0];
B = [-0.3; 0; -17; 0];
C = [0 1 0 0; 0 0 1 0];
D = [0; 0];

% Continuous-time system
sys = ss(A, B, C, D);
disp('Eigenvalues of the continuous system:');
disp(eig(sys));

% Define initial state
x0 = [0; 0; 0; 100]; % Initial conditions for states

% Simulate the continuous-time system response with initial conditions only
t = 0:0.01:10; % Time vector for simulation
[~, t_cont, x_cont] = initial(sys, x0, t);

% Plot x2 and x4 to observe the response
figure;
plot(t_cont, x_cont(:,2), 'DisplayName', 'x2'); hold on;
plot(t_cont, x_cont(:,4), 'DisplayName', 'x4');
xlabel('Time (s)');
ylabel('State values');
legend;
title('Continuous-time System Response with Initial Conditions');

% Discretize the system
Ts = 0.25; % Sampling time
d_sys = c2d(sys, Ts);

% Step response of the discrete-time system
figure
step(d_sys);

% Define Q and R matrices
Q = eye(4); % Identity matrix of size 4x4
R = 100;

% Solve Discrete Algebraic Riccati Equation (DARE)
[x, l, g] = dare(d_sys.A, d_sys.B, Q, R);
disp('Solution to DARE (x):');
disp(x);
disp('LQR gain (l):');
disp(l);
disp('Gain matrix (g):');
disp(g);

% Design LQR controller for continuous system
K = lqr(sys, Q, R);

% Closed-loop continuous-time system
sys_cl_cont = ss(A - B * K, B, C, D);

% Simulate the continuous-time closed-loop system response for x4 and x2
t = 0:0.01:10; % Time vector
[~, t_cont, x_cont] = initial(sys_cl_cont, x0, t);

figure;
plot(t_cont, x_cont(:,2), 'DisplayName', 'x2'); hold on;
plot(t_cont, x_cont(:,4), 'DisplayName', 'x4');
xlabel('Time (s)');
ylabel('State values');
legend;
title('Continuous-time Closed-loop System Response for x2 and x4');

% Discrete LQR controller
K_dis = lqr(d_sys, Q, R);

% Closed-loop discrete-time system
sys_cl_dis = ss(d_sys.A - d_sys.B * K_dis, d_sys.B, d_sys.C, d_sys.D, Ts);

% Simulate the discrete-time closed-loop system response
N = 40; % Number of steps
x_dis = zeros(4, N+1);
u_dis = zeros(1, N);
udot_dis = zeros(1, N);
x_dis(:,1) = x0;

for k = 1:N
    % Calculate control input u
    u_dis(k) = -K_dis * x_dis(:,k);
    
    % Update state
    x_dis(:,k+1) = d_sys.A * x_dis(:,k) + d_sys.B * u_dis(k);
    
    % Calculate derivative of control input (udot)
    if k > 1
        udot_dis(k) = (u_dis(k) - u_dis(k-1)) / Ts;
    end
end

% Plot x2, u, and u_dot in discrete time
time_dis = (0:N) * Ts;

figure;
subplot(3,1,1);
stem(time_dis, x_dis(2,:), 'DisplayName', 'x2');
xlabel('Time (s)');
ylabel('x2');
title('Discrete-time Response of x2');

subplot(3,1,2);
stem(time_dis(1:end-1), u_dis, 'DisplayName', 'u');
xlabel('Time (s)');
ylabel('Control Input (u)');
title('Discrete-time Control Input (u)');

subplot(3,1,3);
stem(time_dis(1:end-1), udot_dis, 'DisplayName', 'u_dot');
xlabel('Time (s)');
ylabel('Derivative of Control Input (u_dot)');
title('Discrete-time Derivative of Control Input (u_dot)');

