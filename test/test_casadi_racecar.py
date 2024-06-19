import casadi as ca
import numpy as np

class CasadiMPCController:
    def __init__(self, dt, N, Q, R, state_bounds, control_bounds):
        self.dt = dt
        self.N = N
        self.Q = Q
        self.R = R
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds

        self.nx = 4  
        self.nu = 2 

        # Define the model equations
        self.model = self._create_model()

        # Create the MPC solver
        self.solver = self._create_solver()

    def _create_model(self):
        # Define the states
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        yaw = ca.SX.sym('yaw')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y, yaw, v)

        # Define the controls
        a = ca.SX.sym('a')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(a, delta)

        # Define the model equations
        rhs = ca.vertcat(
            v * ca.cos(yaw),
            v * ca.sin(yaw),
            v * ca.tan(delta) / 0.325, 
            a
        )

        # Create the model
        model = ca.Function('model', [states, controls], [rhs], ['x', 'u'], ['rhs'])

        return model

    def _create_solver(self):
        # States
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        v = ca.SX.sym('v')
        states = ca.vertcat(x, y, theta, v)
        n_states = states.numel()

        # Controls
        a = ca.SX.sym('a')
        delta = ca.SX.sym('delta')
        controls = ca.vertcat(a, delta)
        n_controls = controls.numel()

        # State and control trajectories
        X = ca.SX.sym('X', n_states, self.N + 1)
        X_ref = ca.SX.sym('X_ref', n_states, self.N + 1)
        U = ca.SX.sym('U', n_controls, self.N)
        U_ref = ca.SX.sym('U_ref', n_controls, self.N)

        # Cost function
        cost_fn = 0
        for k in range(self.N):
            st_err = X[:, k] - X_ref[:, k]
            con_err = U[:, k] - U_ref[:, k]
            cost_fn += ca.mtimes(st_err.T, self.Q) @ st_err
            cost_fn += ca.mtimes(con_err.T, self.R) @ con_err

        # Constraints
        g = [X[:, 0] - X_ref[:, 0]]
        for k in range(self.N):
            k1 = self.model(X[:, k], U[:, k])
            k2 = self.model(X[:, k] + (self.dt / 2) * k1, U[:, k])
            k3 = self.model(X[:, k] + (self.dt / 2) * k2, U[:, k])
            k4 = self.model(X[:, k] + self.dt * k3, U[:, k])
            x_next = X[:, k] + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            g.append(X[:, k + 1] - x_next)

        cost_fn = cost_fn + ca.mtimes((X[:, N] - X_ref[:, N]).T, self.Q) @ (X[:, N] - X_ref[:, N])

        # Optimization variables
        opt_vars = ca.vertcat(
            X.reshape((-1, 1)),
            U.reshape((-1, 1))
        )

        # Optimization parameters
        opt_params = ca.vertcat(
            X_ref.reshape((-1, 1)),
            U_ref.reshape((-1, 1))
        )

        # NLP problem
        nlp_prob = {
            'f': cost_fn,
            'x': opt_vars,
            'g': ca.vertcat(*g),
            'p': opt_params
        }

        # NLP solver options
        opts = {
            'ipopt.max_iter': 5000,
            'ipopt.print_level': 0,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-4,
            'print_time': 0
        }

        # Create NLP solver
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        return solver

    
    def update_state(self, x, u):
        k1 = self.model(x, u)
        k2 = self.model(x + (self.dt / 2) * k1, u)
        k3 = self.model(x + (self.dt / 2) * k2, u)
        k4 = self.model(x + self.dt * k3, u)

        x_next = x + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

        return x_next

if __name__ == "__main__":

    dt = 0.01
    N = 20
    Q = np.diag([10, 10, 1, 1])
    R = np.diag([0.1, 0.1])
    state_bounds = np.array([[-10, 10], [-10, 10], [-np.pi, np.pi], [-5, 5]])
    control_bounds = np.array([[-1, 1], [-0.5, 0.5]])

    mpc_controller = CasadiMPCController(dt, N, Q, R, state_bounds, control_bounds)

    # Simulate the system
    simulation_steps = int(20 / dt)  
    current_state = np.array([0, 0, 0, 0])
    target_state = np.array([5, 5, 0, 0])
    control_ref = np.array([1, 0])

    # Prepare variables for optimization
    lbx = ca.DM.zeros((mpc_controller.nx * (N + 1) + mpc_controller.nu * N, 1))
    ubx = ca.DM.zeros((mpc_controller.nx * (N + 1) + mpc_controller.nu * N, 1))

    lbx[0: mpc_controller.nx * (N + 1):mpc_controller.nx] = -10
    lbx[1: mpc_controller.nx * (N + 1):mpc_controller.nx] = -10
    lbx[2: mpc_controller.nx * (N + 1):mpc_controller.nx] = -np.pi
    lbx[3: mpc_controller.nx * (N + 1):mpc_controller.nx] = -5

    ubx[0: mpc_controller.nx * (N + 1):mpc_controller.nx] = 10
    ubx[1: mpc_controller.nx * (N + 1):mpc_controller.nx] = 10
    ubx[2: mpc_controller.nx * (N + 1):mpc_controller.nx] = np.pi
    ubx[3: mpc_controller.nx * (N + 1):mpc_controller.nx] = 5

    lbx[mpc_controller.nx * (N + 1)  : mpc_controller.nx * (N + 1) + mpc_controller.nu * N: mpc_controller.nu ] = control_bounds[0, 0]
    lbx[mpc_controller.nx * (N + 1)+1: mpc_controller.nx * (N + 1) + mpc_controller.nu * N: mpc_controller.nu ] = control_bounds[1, 0]

    ubx[mpc_controller.nx * (N + 1)  : mpc_controller.nx * (N + 1) + mpc_controller.nu * N: mpc_controller.nu ] = control_bounds[0, 1]
    ubx[mpc_controller.nx * (N + 1)+1: mpc_controller.nx * (N + 1) + mpc_controller.nu * N: mpc_controller.nu ] = control_bounds[1, 1]

    lbg = ca.DM.zeros((mpc_controller.nx * (N + 1), 1))
    ubg = ca.DM.zeros((mpc_controller.nx * (N + 1), 1))

    for _ in range(simulation_steps):
        X_ref = ca.repmat(target_state, 1, N + 1)
        U_ref = ca.repmat(control_ref, 1, N)

        x0 = ca.vertcat(
            ca.repmat(current_state, 1, N + 1).reshape((-1, 1)),
            ca.DM.zeros((mpc_controller.nu * N, 1))
        )

        p = ca.vertcat(
            X_ref.reshape((-1, 1)),
            U_ref.reshape((-1, 1))
        )

        sol = mpc_controller.solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            lbg=lbg,
            ubg=ubg,
            p=p
        )

        X_opt = ca.reshape(sol['x'][:mpc_controller.nx * (N + 1)], mpc_controller.nx, N + 1)
        U_opt = ca.reshape(sol['x'][mpc_controller.nx * (N + 1):], mpc_controller.nu, N)

        # Extract the first optimal control input
        optimal_control = np.array(U_opt[:, 0].full()).flatten()

        # Apply the optimal control to the system and update the state
        current_state = mpc_controller.update_state(current_state, optimal_control)

        # Print the current state
        print("Current state:", current_state)

    print("Final state:", current_state)