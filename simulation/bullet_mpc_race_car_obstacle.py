import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import casadi as ca
import pybullet as p
import pybullet_data
import math

from scipy.linalg import block_diag
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from models.raceCarSim import RacecarModel
from typing import Tuple

from pynput import keyboard

terminate_simulation = False

def on_press(key):
    global terminate_simulation
    try:
        if key.char == 'q':
            terminate_simulation = True
    except AttributeError:
        pass

def plot_arrow(x, y, yaw, length=0.05, width=0.3, fc="b", ec="k"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
            fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def export_casadi_model():

    L = 0.325
    ## Define the states
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    yaw = ca.SX.sym('yaw')
    v = ca.SX.sym('v')
    states = ca.vertcat(x, y, yaw, v)

    ## Define the inputs
    a = ca.SX.sym('a')
    delta = ca.SX.sym('delta')
    controls = ca.vertcat(a, delta)

    ## Define the system equations \dot(x) = f(x,u)
    rhs = ca.vertcat(
        v * ca.cos(yaw),
        v * ca.sin(yaw),
        v * ca.tan(delta) / L,
        a
    )


    ## Define implicit DAE
    xdot = ca.SX.sym('xdot')
    ydot = ca.SX.sym('ydot')
    yawdot = ca.SX.sym('yawdot')
    vdot = ca.SX.sym('vdot')
    dae = ca.vertcat(xdot, ydot, yawdot, vdot)

    ## Define dynamics system
    ### Explicit
    f_expl = rhs
    ### Implicit
    f_impl = dae - rhs

    # set model
    model = AcadosModel()
    model.x = states
    model.u = controls
    model.z = []
    # model.p = ca.SX.sym('p', 6)
    model.xdot = dae
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.name = "RaceCarObstacle"

    return model


class MPCController:

    def __init__(
            self,
            x0: np.ndarray,
            u0: np.ndarray,
            # goal_trajectory: np.ndarray,
            # goal_control: np.ndarray,
            state_cost_matrix: np.ndarray,
            control_cost_matrix: np.ndarray,
            terminal_cost_matrix: np.ndarray,
            state_lower_bound: np.ndarray,
            state_upper_bound: np.ndarray,
            control_lower_bound: np.ndarray,
            control_upper_bound: np.ndarray,
            obstacle_positions: np.ndarray,
            obstacle_radii: np.ndarray,
            safe_distance: float,
            N: int,
            dt: float,
            Ts: float,
            cost_type: str
              
        ) -> None:
        """
        Initialize the MPC controller with the given parameters.
        """

        # Cost type
        self.cost_type = cost_type

        self.state_cost_matrix = state_cost_matrix
        self.control_cost_matrix = control_cost_matrix
        self.terminal_const_matrix = terminal_cost_matrix

        self.state_constraints = {
            'lbx': state_lower_bound,
            'ubx': state_upper_bound,
        }

        self.control_constraints = {
            'lbu': control_lower_bound,
            'ubu': control_upper_bound
        }

        self.obstacle_positions = obstacle_positions
        self.obstacle_radii     = obstacle_radii
        self.safe_distance      = safe_distance

        self.N = N
        self.dt = dt
        self.Ts = Ts

        # RaceCar  Model
        self.racecar = RacecarModel(x0, L=0.325, dt=dt)

        # Export the Casadi Model
        self.model = export_casadi_model()

        # Create the ACADOS solver
        self.mpc_solver, self.mpc = self.create_mpc_solver(x0)

        # Create the ACADOS simulator
        # self.sim_solver = self.create_sim_solver()

    def create_mpc_solver(self, x0: np.ndarray) -> Tuple[AcadosOcpSolver, AcadosOcp]:
        mpc = AcadosOcp()

        model = self.model
        mpc.model = model
        nx = model.x.size()[0]
        nu = model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # set dimensions
        mpc.dims.N = self.N
        mpc.dims.nx = nx
        mpc.dims.nu = nu

        # Set cost
        Q_mat = self.state_cost_matrix
        Q_mat_e = self.terminal_const_matrix
        R_mat = self.control_cost_matrix    

        # External cost function
        if self.cost_type == 'LINEAR_LS':
            unscale = self.N / self.Ts
            mpc.cost.cost_type = 'LINEAR_LS'
            mpc.cost.cost_type_e = 'LINEAR_LS'
            mpc.cost.W = block_diag(Q_mat, R_mat)
            mpc.cost.W_e = Q_mat_e
            Vx = np.zeros((ny, nx))
            Vx[:nx, :nx] = np.eye(nx)
            mpc.cost.Vx = Vx
            mpc.cost.Vx_e = np.eye(nx)
            Vu = np.zeros((ny, nu))
            Vu[nx : nx + nu, 0:nu] = np.eye(nu)
            mpc.cost.Vu = Vu
            mpc.cost.yref = np.zeros((ny, ))
            mpc.cost.yref_e = np.zeros((ny_e, ))

        elif self.cost_type == 'NONLINEAR_LS':
            mpc.cost.cost_type = 'NONLINEAR_LS'
            mpc.cost.cost_type_e = 'NONLINEAR_LS'
            mpc.model.cost_y_expr = ca.vertcat(model.x, model.u)
            mpc.model.cost_y_expr_e = model.x
            mpc.cost.yref = np.zeros((ny, ))
            mpc.cost.yref_e = np.zeros((ny_e, ))
            mpc.cost.W = block_diag(Q_mat, R_mat)
            mpc.cost.W_e = Q_mat_e

        

        # Set constraints
        mpc.constraints.lbx = self.state_constraints['lbx']
        mpc.constraints.ubx = self.state_constraints['ubx']
        mpc.constraints.idxbx = np.arange(nx)

        mpc.constraints.lbx_e = self.state_constraints['lbx']
        mpc.constraints.ubx_e = self.state_constraints['ubx']
        mpc.constraints.idxbx_e = np.arange(nx)

        mpc.constraints.x0  = x0

        mpc.constraints.lbu = self.control_constraints['lbu']
        mpc.constraints.ubu = self.control_constraints['ubu']
        mpc.constraints.idxbu = np.arange(nu)

        # Define obstacle avoidance constraints
        # num_obstacles = len(self.obstacle_positions)
        # mpc.constraints.lh = np.zeros((num_obstacles,))
        # mpc.constraints.uh = np.full((num_obstacles,), 1e8)

        # obstacle_pos_sym = ca.SX.sym('obstacle_pos', 2, num_obstacles)

        # mpc.model.p = ca.vertcat(mpc.model.p, ca.vec(obstacle_pos_sym))
        # mpc.model.con_h_expr = ca.vertcat(*[
        #     (model.x[0] - obstacle_pos_sym[0, i])**2 + (model.x[1] - obstacle_pos_sym[1, i])**2 - (self.obstacle_radii[i] + self.safe_distance)**2
        #     for i in range(num_obstacles)
        # ])
        # mpc.parameter_values = np.zeros(mpc.model.p.size()[0])

        # for i in range(num_obstacles):
        #     mpc.constraints.lh[i] = (self.obstacle_radii[i] + self.safe_distance)**2

        # Set solver options
        mpc.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'ERK'
        mpc.solver_options.nlp_solver_type = 'SQP_RTI'
        mpc.solver_options.sim_method_num_stages = 4
        mpc.solver_options.sim_method_num_steps = 3
        mpc.solver_options.nlp_solver_max_iter = 100
        mpc.solver_options.qp_solver_cond_N = self.N

        # Set prediction horizons
        mpc.solver_options.tf = self.Ts

        try:
            print("Creating MPC solver...")
            mpc_solver = AcadosOcpSolver(mpc, json_file='race_car_obstacle_mpc.json')
            print("MPC solver created successfully.")
            return mpc_solver, mpc
        except Exception as e:
            print("Error creating MPC solver:", e)
            raise

    def create_sim_solver(self) -> AcadosSimSolver:
        sim = AcadosSim()

        model = self.model
        sim.model = model

        # Set simulation options
        sim.solver_options.T = self.dt
        sim.solver_options.num_stages = 4
        sim.solver_options.num_steps = 3

        try:
            print("Creating Sim solver...")
            sim_solver = AcadosSimSolver(sim, json_file='race_car_obstacle_sim.json')
            print("Sim solver created successfully.")
            return sim_solver
        except Exception as e:
            print("Error creating Sim solver:", e)
            raise



    def solve_mpc(self, x0: np.ndarray,
                        simX: np.ndarray,
                        simU: np.ndarray,
                        yref: np.ndarray,
                        yref_N: np.ndarray,
                        obstacle_positions: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the MPC problem with the given initial state.
        """

        nx = self.model.x.size()[0]

        # Ensure x0 is the correct shape
        if x0.shape[0] != nx:
            raise ValueError(f"Initial state x0 must have dimension {nx}, but got {x0.shape[0]}.")

        # Update the initial state in the solver
        self.mpc_solver.set(0, "lbx", x0)
        self.mpc_solver.set(0, "ubx", x0)

        # Set the obstacle positions
        # param_values = np.concatenate((np.zeros(6), obstacle_positions.flatten()))

        # # Set the obstacle positions in the solver
        # for i in range(self.N+1):
        #     self.mpc_solver.set(i, 'p', param_values)

        # Set reference trajectory
        for i in range(self.mpc.dims.N):
            self.mpc_solver.set(i, "yref", yref)
        self.mpc_solver.set(self.mpc.dims.N, "yref", yref_N)

        # Warm up the solver
        for i in range(self.N):
            self.mpc_solver.set(i, "x", simX[i, :])
            self.mpc_solver.set(i, "u", simU[i, :])
        self.mpc_solver.set(self.N, "x", simX[self.N, :])

        # Solve the MPC problem
        status = self.mpc_solver.solve()

        if status != 0:
            # raise Exception(f'Acados returned status {status}')
            print("Solver failed with status: ", status)

        for i in range(self.mpc.dims.N):
            self.mpc_solver.set(i, "yref", yref)
            simX[i, :] = self.mpc_solver.get(i, "x")
            simU[i, :] = self.mpc_solver.get(i, "u")
        simX[self.mpc.dims.N, :] = self.mpc_solver.get(self.mpc.dims.N, "x")

        return simX, simU
    
    def runge_kutta(self, f: np.ndarray, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute the state update using the Runge-Kutta method.
        """

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)

        return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    

    def update_stateRungeKutta(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Use the Runge-Kutta method to update the state based on the current state and control inputs.
        """

        x1 = self.runge_kutta(self.racecar.forward_kinematic, x0, u, self.dt)
        return x1


    def update_state(self, x0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Use the AcadosSimSolver to update the state based on the current state and control inputs.
        """

        self.sim_solver.set("x", x0)
        self.sim_solver.set("u", u)
        status = self.sim_solver.solve()

        if status != 0:
            # raise Exception(f'Simulation failed with status {status}')
            print("Solver failed with status: ", status)

        x1 = self.sim_solver.get('x')
        return x1
    
def key_listener():
    """
    Simple key listener function to detect when 'Q' is pressed.
    """
    import msvcrt  # Windows only
    if msvcrt.kbhit():
        key = msvcrt.getch()
        if key == b'q' or key == b'Q':
            return True
    return False

def inverse_kinematic(v, delta, L, W):
    omega = v * math.tan(delta) / L

    v_lrw = v * (1 - W * math.tan(delta) / (2 * L))
    v_rrw = v * (1 + W * math.tan(delta) / (2 * L))

    v_lfw = math.sqrt(v**2 + ((v * math.tan(delta) / 2) - (v * W * math.tan(delta) / (2 * L)))**2)
    v_rfw = math.sqrt(v**2 + ((v * math.tan(delta) / 2) + (v * W * math.tan(delta) / (2 * L)))**2)

    return v_lrw, v_rrw, v_lfw, v_rfw

if __name__ == "__main__":
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load the plane and the race car
    plane_id = p.loadURDF("plane.urdf")
    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    car_id = p.loadURDF("/home/eroxii/ocp_ws/RL-MPPI-MPC/urdf/racecar/racecar.urdf", start_pos, start_orientation)


    # Get the joint information
    num_joints = p.getNumJoints(car_id)
    steering_joints = []
    drive_joints = []

    for i in range(num_joints):
        joint_info = p.getJointInfo(car_id, i)
        joint_name = joint_info[1].decode('utf-8')

        if 'steering' in joint_name:
            steering_joints.append(i)
        elif joint_name == 'left_rear_wheel_joint' or joint_name == 'right_rear_wheel_joint' or \
            joint_name == 'left_front_wheel_joint' or joint_name == 'right_front_wheel_joint':
            drive_joints.append(i)
    

    # Set up the MPC controller
    state_init = np.array([0.0, 0.0, 0.0, 0.0])
    control_init = np.array([0.0, 0.0])

    ## Cost Matrices
    state_cost_matrix = np.diag([55, 55, 75, 75]) 
    control_cost_matrix = np.diag([0.1, 1])                
    terminal_cost_matrix = np.diag([55, 55, 75, 75])  

    ## Constraints
    state_lower_bound = np.array([-50.0, -50.0, -np.pi, -10.0])
    state_upper_bound = np.array([50.0, 50.0, np.pi, 10.0])
    control_lower_bound = np.array([-5.0, -3.14])  
    control_upper_bound = np.array([5.0, 3.14])

    obstacle_positions = np.array([[-2.0, -2.0]])  
    obstacle_radii = np.array([0.5])
    safe_distance = 0.2

    N = 100
    dt = 0.01
    Ts = 3.0

    mpc_controller = MPCController(
        x0=state_init,
        u0=control_init,
        state_cost_matrix=state_cost_matrix,
        control_cost_matrix=control_cost_matrix,
        terminal_cost_matrix=terminal_cost_matrix,
        state_lower_bound=state_lower_bound,
        state_upper_bound=state_upper_bound,
        control_lower_bound=control_lower_bound,
        control_upper_bound=control_upper_bound,
        obstacle_positions=obstacle_positions,
        obstacle_radii=obstacle_radii,
        safe_distance=safe_distance,
        N=N,
        dt=dt,
        Ts=Ts,
        cost_type='LINEAR_LS'
    )

    # Enable real-time simulation
    p.setRealTimeSimulation(1)

    # Simulation loop
    velocity = 0.0

    simX = np.zeros((mpc_controller.N + 1, mpc_controller.model.x.size()[0]))
    simU = np.zeros((mpc_controller.N, mpc_controller.model.u.size()[0]))

    goal_position = np.array([1.0, 1.0])  # Example goal position
    goal_threshold = 0.1  # Threshold distance to consider the goal reached

    while not terminate_simulation:
        # Get the current state of the race car
        pos, ori = p.getBasePositionAndOrientation(car_id)
        vel, ang_vel = p.getBaseVelocity(car_id)
        euler = p.getEulerFromQuaternion(ori)
        yaw = euler[2]
        state_current = np.array([pos[0], pos[1], yaw, vel[0]])

        # Check if the goal is reached
        if np.linalg.norm(state_current[:2] - goal_position) < goal_threshold:
            print("Goal reached")
            # break

        # Get the reference trajectory and control input
        state_ref = np.array([2.0, 2.0, 0.0, 0.0]) 
        control_ref = np.array([0.0, 0.0]) 
        yref = np.concatenate((state_ref, control_ref))
        yref_N = state_ref

        # Solve the MPC problem
        simX, simU = mpc_controller.solve_mpc(state_current, simX, simU, yref, yref_N, obstacle_positions)

        # Get the optimal control input
        control_input = simU[0, :]

        # Apply the control input to the race car
        target_acceleration = control_input[0]
        target_steering = control_input[1]

        # Update the velocity based on the acceleration
        velocity += target_acceleration * dt

        # Apply the wheel velocities to the rear wheels
        for j in drive_joints:
            p.setJointMotorControl2(car_id, j, p.VELOCITY_CONTROL, targetVelocity=velocity, force=10)

        # p.setJointMotorControl2(car_id, drive_joints[0], p.VELOCITY_CONTROL, targetVelocity=v_lrw, force=10)  # Left rear wheel
        # p.setJointMotorControl2(car_id, drive_joints[1], p.VELOCITY_CONTROL, targetVelocity=v_rrw, force=10)  # Right rear wheel
        # p.setJointMotorControl2(car_id, drive_joints[2], p.VELOCITY_CONTROL, targetVelocity=v_lfw, force=10)  # Left front wheel
        # p.setJointMotorControl2(car_id, drive_joints[3], p.VELOCITY_CONTROL, targetVelocity=v_rfw, force=10)

        # Apply the steering angles to the front wheels
        for j in steering_joints:
            p.setJointMotorControl2(car_id, j, p.POSITION_CONTROL, targetPosition=target_steering)

        print(f"State = {state_current}, Control = {control_input}\n")

        # Step the simulation
        p.stepSimulation()
        time.sleep(dt)

    # Disconnect from PyBullet
    p.disconnect()