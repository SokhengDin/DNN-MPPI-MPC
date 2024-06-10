import casadi as ca
import numpy as np

class CasadiMPCController:
    def __init__(self, dt, N, Q, R, Qf, state_bounds, control_bounds):
        self.dt = dt
        self.N = N
        self.Q = Q
        self.R = R
        self.Qf = Qf
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds

        self.nx = 4  # Number of states
        self.nu = 2  # Number of controls

        # Define the model equations
        self.model = self._create_model()

        # Create the MPC solver
        self.solver, self.args = self._create_solver()

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
        # Define the decision variables
        x = ca.SX.sym('x', self.nx, self.N + 1)
        u = ca.SX.sym('u', self.nu, self.N)

        # Define the reference trajectory parameter
        x_ref = ca.SX.sym('x_ref', self.nx, self.N + 1)

        # Define the initial state parameter
        x0 = ca.SX.sym('x0', self.nx)

        # Define the cost function
        cost_fn = 0
        constraints = []
        constraints.append(x[:, 0] - x0)

        for k in range(self.N):
            cost_fn += ca.mtimes(ca.mtimes((x[:, k] - x_ref[:, k]).T, self.Q), (x[:, k] - x_ref[:, k]))
            cost_fn += ca.mtimes(ca.mtimes(u[:, k].T, self.R), u[:, k])

            # Runge-Kutta 4th order integration
            k1 = self.model(x[:, k], u[:, k])
            k2 = self.model(x[:, k] + self.dt / 2 * k1, u[:, k])
            k3 = self.model(x[:, k] + self.dt / 2 * k2, u[:, k])
            k4 = self.model(x[:, k] + self.dt * k3, u[:, k])
            x_next = x[:, k] + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            constraints.append(x_next - x[:, k + 1])

        # Terminal cost
        cost_fn += ca.mtimes(ca.mtimes((x[:, self.N] - x_ref[:, self.N]).T, self.Qf), (x[:, self.N] - x_ref[:, self.N]))

        # Set up the NLP problem
        nlp_prob = {
            'f': cost_fn,
            'x': ca.vertcat(ca.reshape(x, -1, 1), ca.reshape(u, -1, 1)),
            'g': ca.vertcat(*constraints),
            'p': ca.vertcat(x0, ca.reshape(x_ref, -1, 1))
        }

        # Set up the NLP solver options
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

        # Create the NLP solver
        nlp_solver = ca.nlpsol('nlp_solver', 'ipopt', nlp_prob, opts)

        # Set up the lower and upper bounds for decision variables
        lbx = ca.vertcat(
            ca.reshape(ca.repmat(self.state_bounds[:, 0], self.N + 1), -1, 1),
            ca.reshape(ca.repmat(self.control_bounds[:, 0], self.N), -1, 1)
        )
        ubx = ca.vertcat(
            ca.reshape(ca.repmat(self.state_bounds[:, 1], self.N + 1), -1, 1),
            ca.reshape(ca.repmat(self.control_bounds[:, 1], self.N), -1, 1)
        )

        # Set up the arguments for the solver
        args = {
            'lbx': lbx,
            'ubx': ubx,
            'lbg': ca.DM.zeros(self.nx * (self.N + 1), 1),
            'ubg': ca.DM.zeros(self.nx * (self.N + 1), 1)
        }

        return nlp_solver, args

    def solve(self, initial_state, target_state):
        # Set the reference trajectory
        x_ref = ca.repmat(target_state, 1, self.N + 1)

        # Set the initial state
        x0 = initial_state

        # Set the initial guess for decision variables
        x_guess = ca.repmat(initial_state, 1, self.N + 1)
        u_guess = ca.DM.zeros((self.nu, self.N))
        guess = ca.vertcat(ca.reshape(x_guess, -1, 1), ca.reshape(u_guess, -1, 1))

        # Solve the NLP problem
        sol = self.solver(x0=guess, p=ca.vertcat(x0, ca.reshape(x_ref, -1, 1)), **self.args)

        # Extract the optimal control inputs
        u_opt = sol['x'][self.nx * (self.N + 1):]

        return ca.reshape(u_opt, (self.nu, self.N))[:, 0]


if __name__ == "__main__":
    # Set up the MPC controller
    dt = 0.01
    N = 20
    Q = np.diag([10, 10, 10, 1])  # Increased weight on state tracking
    R = np.diag([0.1, 0.1])
    Qf = np.diag([100, 100, 100, 10])  # Increased terminal cost
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

    mpc_controller = CasadiMPCController(dt, N, Q, R, Qf, state_bounds, control_bounds)

    # Simulate the system
    simulation_steps = 100
    current_state = np.array([0, 0, 0, 0])
    target_state = np.array([5, 5, 0, 0])

    for _ in range(simulation_steps):
        # Solve the MPC problem
        optimal_control = mpc_controller.solve(current_state, target_state)

        # Apply the optimal control input to the system
        current_state = mpc_controller.model(current_state, optimal_control).full().flatten()

        print("Current state:", current_state)

    print("Final state:", current_state)