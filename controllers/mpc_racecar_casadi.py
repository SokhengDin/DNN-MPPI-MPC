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

        self.nx = 4  # Number of states
        self.nu = 2  # Number of controls

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
            v * ca.tan(delta) / 0.325,  # Wheelbase length
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
            cost_fn += st_err.T @ self.Q @ st_err
            cost_fn += con_err.T @ self.R @ con_err

        # Terminal cost
        st_err = X[:, self.N] - X_ref[:, self.N]
        cost_fn += st_err.T @ self.Q @ st_err

        # Constraints
        g = [X[:, 0] - X_ref[:, 0]]
        for k in range(self.N):
            k1 = self.model(X[:, k], U[:, k])
            k2 = self.model(X[:, k] + self.dt/2 * k1, U[:, k])
            k3 = self.model(X[:, k] + self.dt/2 * k2, U[:, k])
            k4 = self.model(X[:, k] + self.dt * k3, U[:, k])
            x_next = X[:, k] + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            g.append(x_next - X[:, k+1])

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
            'ipopt.print_level': 0,
            'ipopt.max_iter': 1000,
            'ipopt.tol': 1e-8,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.mu_oracle': 'quality-function',
            'ipopt.bound_frac': 0.001,
            'ipopt.bound_push': 0.001,
            'print_time': 0
        }


        # Create NLP solver
        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        return solver

    def solve(self, initial_state, target_state, control_ref):
        # Set initial state and target state
        X_ref = ca.repmat(ca.vertcat(target_state), 1, self.N + 1)
        U_ref = ca.repmat(ca.vertcat(control_ref), 1, self.N)

        # Set lower and upper bounds for states and controls
        lbx = ca.vertcat(
            ca.repmat(self.state_bounds[:, 0], self.N + 1),
            ca.repmat(self.control_bounds[:, 0], self.N)
        )
        ubx = ca.vertcat(
            ca.repmat(self.state_bounds[:, 1], self.N + 1),
            ca.repmat(self.control_bounds[:, 1], self.N)
        )

        # Set initial guess for optimization variables
        x0 = ca.vertcat(
            ca.repmat(initial_state, 1, self.N + 1).reshape((-1, 1)),
            ca.repmat(control_ref, 1, self.N).reshape((-1, 1))
        )

        # Solve the NLP problem
        sol = self.solver(x0=x0, lbx=lbx, ubx=ubx, p=ca.vertcat(X_ref.reshape((-1, 1)), U_ref.reshape((-1, 1))))

        # Extract optimal control input
        u_opt = ca.reshape(sol['x'][-self.N * self.nu:], self.nu, self.N)

        return u_opt[:, 0]


# Example usage in main function
if __name__ == "__main__":
    # Set up the MPC controller
    dt = 0.1
    N = 20
    Q = np.diag([10, 10, 10, 10])
    R = np.diag([0.1, 0.1])
    state_bounds = np.array([
        [-10, 10],
        [-10, 10],
        [-np.pi, np.pi],
        [-5, 5]
    ])
    control_bounds = np.array([
        [-1, 1],
        [-0.5, 0.5]
    ])

    mpc_controller = CasadiMPCController(dt, N, Q, R, state_bounds, control_bounds)

    # Simulate the system
    simulation_steps = 100
    current_state = np.array([0, 0, 0, 0])
    target_state = np.array([5, 5, 0, 0])
    control_ref = np.array([0.0, 0])

    for _ in range(simulation_steps):
        # Solve the MPC problem
        optimal_control = mpc_controller.solve(current_state, target_state, control_ref)

        # Check if solver failed
        if optimal_control is None:
            break

        # Update the current state using the model equations
        next_state = mpc_controller.model(current_state, optimal_control).full().flatten()
        current_state = current_state + mpc_controller.dt * next_state

        print("Current state:", current_state)

    print("Final state:", current_state)
