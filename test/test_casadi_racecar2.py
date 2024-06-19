import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import math

# Constants
N = 100
step_time = 0.01
sim_time = 30

# State and control limits
x_min, y_min, yaw_min, v_min = -ca.inf, -ca.inf, -ca.inf, -ca.inf
x_max, y_max, yaw_max, v_max = ca.inf, ca.inf, ca.inf, ca.inf
a_min, a_max = -1.0, 1.0
delta_min, delta_max = -0.5, 0.5

# Generate reference trajectory
def simple_trajectory_generator(t):
    a = 2.0  # Adjust this value to change the size of the lemniscate
    x_ref = a * np.cos(t) / (1 + np.sin(t)**2)
    y_ref = a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)
    yaw_ref = np.arctan2(np.cos(t) * (np.cos(t)**2 - 2*np.sin(t)**2), (1 + np.sin(t)**2)**2)
    v_ref = a * np.sqrt(2) * np.sqrt((np.cos(t)**2 + np.sin(t)**2) / (1 + np.sin(t)**2)**2)

    return x_ref, y_ref, yaw_ref, v_ref

# Plot arrow function
def plot_arrow(x, y, yaw, length=0.5, width=0.25, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def forward_kinematic(x, u):
    vx = x[3] * ca.cos(x[2])
    vy = x[3] * ca.sin(x[2])
    vyaw = x[3] * ca.tan(u[1]) / 0.325
    a  = u[0]

    x_next = np.array([vx, vy, vyaw, a])

    return x_next

def runge_kutta(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + (dt/2) *k1, u)
    k3 = f(x + (dt/2) *k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)  

    return x_next 

# Define states and controls
x = ca.SX.sym('x')
y = ca.SX.sym('y')
yaw = ca.SX.sym('yaw')
v = ca.SX.sym('v')
states = ca.vertcat(x, y, yaw, v)
n_states = states.numel()

a = ca.SX.sym('a')
delta = ca.SX.sym('delta')
controls = ca.vertcat(a, delta)
n_controls = controls.numel()

# Define cost function weights
Q = 1000*np.diag([1.0, 1.0, 4.0, 1.5])
R = np.diag([1, 1])

# Define MPC problem
X = ca.SX.sym('X', n_states, N+1)
X_ref = ca.SX.sym('X_ref', n_states, N+1)
U = ca.SX.sym('U', n_controls, N)
U_ref = ca.SX.sym('U_ref', n_controls, N)

cost_fn = 0.0
g = X[:, 0] - X_ref[:, 0]

RHS = ca.vertcat(
    v * ca.cos(yaw),
    v * ca.sin(yaw),
    v * ca.tan(delta) / 0.325,
    a
)

f = ca.Function('f', [states, controls], [RHS])

for k in range(N):
    st_err = X[:, k] - X_ref[:, k]
    con_err = U[:, k] - U_ref[:, k]
    cost_fn = cost_fn + st_err.T @ Q @ st_err + con_err.T @ R @ con_err

cost_fn = cost_fn + (X[:, N] - X_ref[:, N]).T @ Q @ (X[:, N] - X_ref[:, N])

for k in range(N):
    k1 = f(X[:, k], U[:, k])
    k2 = f(X[:, k] + (step_time / 2) * k1, U[:, k])
    k3 = f(X[:, k] + (step_time / 2) * k2, U[:, k])
    k4 = f(X[:, k] + step_time * k3, U[:, k])
    x_next = X[:, k] + (step_time / 6) * (k1 + 2*k2 + 2*k3 + k4)
    g = ca.vertcat(g, X[:, k+1] - x_next)

opt_dec = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

opt_par = ca.vertcat(ca.reshape(X_ref, -1, 1), ca.reshape(U_ref, -1, 1))

nlp_probs = {
    'f': cost_fn,
    'x': opt_dec,
    'p': opt_par,
    'g': g
}

nlp_opts = {
    'ipopt.max_iter': 2000,
    'ipopt.print_level': 0,
    'print_time': 0,
    'ipopt.acceptable_tol': 1e-8,
    'ipopt.acceptable_obj_change_tol': 1e-6
}

solver = ca.nlpsol('solver', 'ipopt', nlp_probs, nlp_opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = x_min
lbx[1: n_states*(N+1): n_states] = y_min
lbx[2: n_states*(N+1): n_states] = yaw_min
lbx[3: n_states*(N+1): n_states] = v_min

ubx[0: n_states*(N+1): n_states] = x_max
ubx[1: n_states*(N+1): n_states] = y_max
ubx[2: n_states*(N+1): n_states] = yaw_max
ubx[3: n_states*(N+1): n_states] = v_max

lbx[n_states*(N+1)  : n_states*(N+1)+n_controls*N: n_controls] = a_min
lbx[n_states*(N+1)+1: n_states*(N+1)+n_controls*N: n_controls] = delta_min

ubx[n_states*(N+1)  : n_states*(N+1)+n_controls*N: n_controls] = a_max
ubx[n_states*(N+1)+1: n_states*(N+1)+n_controls*N: n_controls] = delta_max

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),
    'lbx': lbx,
    'ubx': ubx
}

# Initial conditions
t0 = 0
x_init = 0.0
y_init = 0.0
yaw_init = 0.0
v_init = 0.0

state_init = np.array([x_init, y_init, yaw_init, v_init], dtype=np.float64)
controls_init = np.array([0.0, 0.0], dtype=np.float64)

current_states = state_init
current_controls = controls_init

next_trajectories = np.tile(current_states.reshape(4, 1), N+1)
next_controls = np.tile(current_controls.reshape(2, 1), N)

states = np.tile(current_states.reshape(4, 1), N+1)
controls = np.tile(current_controls.reshape(2, 1), N)
# Simulation loop
mpciter = 0
plt.figure(figsize=(12, 7))

while mpciter * step_time < sim_time:

    args['p'] = np.concatenate([
        next_trajectories.T.reshape(-1, 1),
        next_controls.T.reshape(-1, 1)
    ])

    args['x0'] = np.concatenate([
        states.T.reshape(-1, 1),
        controls.T.reshape(-1, 1)
    ])

    sol = solver(
        x0=args['x0'],
        p=args['p'],
        lbx=args['lbx'],
        ubx=args['ubx'],
        lbg=args['lbg'],
        ubg=args['ubg']
    )

    sol_x = ca.reshape(sol['x'][:n_states*(N+1)], 4, N+1)
    sol_u = ca.reshape(sol['x'][n_states*(N+1):], 2, N)

    a_opt = sol_u.full()[0, 0]
    delta_opt = sol_u.full()[1, 0]

    # x_next = current_states + ca.DM.full(f(current_states, [a_opt, delta_opt])) * step_time
    x_next = ca.DM.full(f(current_states, [a_opt, delta_opt]))

    current_states = x_next

    current_controls = np.array([a_opt, delta_opt])

    print("Current states", current_states)

    next_trajectories[0, 0] = current_states[0]
    next_trajectories[1, 0] = current_states[1]
    next_trajectories[2, 0] = current_states[2]
    next_trajectories[3, 0] = current_states[3]

    for k in range(N):
        t_predict = t0 + k * step_time
        x_ref, y_ref, yaw_ref, v_ref = simple_trajectory_generator(t_predict)
        next_trajectories[0, k+1] = x_ref
        next_trajectories[1, k+1] = y_ref
        next_trajectories[2, k+1] = yaw_ref
        next_trajectories[3, k+1] = v_ref

    next_controls = np.tile(current_controls.reshape(2, 1), N)
    states = np.tile(current_states.reshape(4, 1), N+1)
    controls = np.tile(current_controls.reshape(2, 1), N)


    plt.clf()
    plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
    plot_arrow(current_states[0], current_states[1], current_states[2])
    plt.plot(next_trajectories[0, :], next_trajectories[1, :], marker=".", color="blue", label="Input Trajectory")
    plt.scatter(sol_x.full()[0, :], sol_x.full()[1, :], marker="*", color="red", label="Predicted value")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.pause(0.001)

    t0 += step_time
    mpciter += 1
