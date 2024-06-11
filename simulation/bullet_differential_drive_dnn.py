import casadi as ca
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c

from scipy.linalg import block_diag
from torchvision import models
from acados_template import AcadosModel, AcadosOcpSolver, AcadosOcp

class ResNet18(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 512)
        self.hidden_layers = nn.ModuleList()
        for i in range(2):
            self.hidden_layers.append(nn.Linear(512, 512))
        self.out_layer = nn.Linear(512, output_dim)
        
        # Model is not trained -- setting output to zero
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)
        
    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension
        x = x.repeat(1, 3, 1, 1)  # Repeat the input along the channel dimension
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the output of ResNet18
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x
    


class DifferentialDriveLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        ## Define the states
        x = ca.MX.sym('x')
        y = ca.MX.sym('y')
        yaw = ca.MX.sym('yaw')
        states = ca.vertcat(x, y, yaw)

        ## Define the inputs
        v = ca.MX.sym('v')
        w = ca.MX.sym('w')
        controls = ca.vertcat(v, w)

        model_input = ca.vertcat(states, controls)
        res_model = self.learned_dyn(model_input)

        xdot = ca.MX.sym('xdot')
        ydot = ca.MX.sym('ydot')
        yawdot = ca.MX.sym('yawdot')
        x_dot = ca.vertcat(xdot, ydot, yawdot)

        f_expl = ca.vertcat(
            v * ca.cos(yaw),
            v * ca.sin(yaw),
            w
        ) + res_model

        # Store to struct
        model = ca.types.SimpleNamespace()
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.z = ca.vertcat([])
        model.p = ca.vertcat([])
        model.f_expl = f_expl
        model.constraints = ca.vertcat([])
        model.name = 'differential_drive_dnn_mpc'

        return model
    

class MPC:
    def __init__(self, model, N, external_shared_lib_dir, external_shared_lib_name):
        self.N = N
        self.model = model
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())
    

    def ocp(self):

        model = self.model

        t_horizons = 1.0
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p



        nx = 3
        nu = 2
        ny = nx + nu
        ny_e = nx

        Q_mat = np.diag([15, 10, 35])
        Q_mat_e = np.diag([15, 10, 35])
        R_mat = np.diag([1, 0.1])

        mpc = AcadosOcp()

        mpc.model = model_ac
        mpc.dims.N = N
        mpc.dims.nx = nx
        mpc.dims.nu = nu
        mpc.dims.ny = ny
        mpc.solver_options.tf = t_horizons

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

        l4c_y_expr = None

        mpc.constraints.x0 = np.array([0.0, 0.0, 0.0])
        mpc.constraints.lbu = np.array([-10, -31.4])
        mpc.constraints.ubu = np.array([10, 31.4])
        mpc.constraints.idxbu = np.array([0, 1])

        mpc.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'ERK'
        mpc.solver_options.nlp_solver_type = 'SQP_RTI'
        mpc.solver_options.sim_method_num_stages = 4
        mpc.solver_options.sim_method_num_steps = 3
        mpc.solver_options.nlp_solver_max_iter = 100
        mpc.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        mpc.solver_options.model_external_shared_lib_name = self.external_shared_lib_name

        return mpc
    
    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.name = model.name

        return model_ac
    


def run():
    N = 100

    input_dim = 5  
    output_dim = 3  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load("saved_models/resnet18_diff.pth", map_location=device))
    model.eval() 

    learned_dyn_model = l4c.L4CasADi(
        model,
        model_expects_batch_dim=True,
        name='learned_dynamics_differential_drive'
    )

    model = DifferentialDriveLearnedDynamics(learned_dyn_model)
    solver = MPC(
        model=model.model(),
        N=N,
        external_shared_lib_dir=learned_dyn_model.shared_lib_dir,
        external_shared_lib_name=learned_dyn_model.name).solver

    x = []
    ts = 1.0 / N

    # Initial state
    x_init = np.array([0.0, 0.0, 0.0])
    xt = x_init

    # State and control references
    state_ref = np.array([5.0, 5.0, np.pi/2])  
    control_ref = np.array([0.0, 0.0]) 

    for i in range(50):
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)

        for j in range(N):
            solver.set(j, "yref", np.concatenate((state_ref, control_ref)))
        solver.set(N, "yref", state_ref[:3])

        solver.solve()

        xt = solver.get(1, "x")
        ut = solver.get(0, "u")

        x.append(xt)

        print(f"Iteration {i+1}: Current State - x: {xt[0]:.2f}, y: {xt[1]:.2f}, yaw: {xt[2]:.2f}")


if __name__ == "__main__":
    run()