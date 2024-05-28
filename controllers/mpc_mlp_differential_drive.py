import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import casadi as cs
import numpy as np
import torch
import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
from dnn.resnet50 import ResNet50
from torchvision import models

import time

COST = 'LINEAR_LS'  # NONLINEAR_LS


class ResNet50Modified(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.features = torch.nn.Sequential(*(list(resnet50.children())[:-1]))  # Exclude the final fully connected layer
        self.fc = torch.nn.Linear(resnet50.fc.in_features, 3)  # Adjust the output layer to produce 3 outputs

    def forward(self, x):
        # Assuming x is a vector of shape [batch_size, 3]
        # Reshape x to [batch_size, 3, 1, 1] to make it compatible with ResNet-50 input
        x = x.view(x.size(0), 3, 1, 1)
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)
        return x

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = torch.nn.Linear(3, 512)
        hidden_layers = [torch.nn.Linear(512, 512) for _ in range(2)]
        self.hidden_layer = torch.nn.ModuleList(hidden_layers)
        self.out_layer = torch.nn.Linear(512, 3)
        with torch.no_grad():
            self.out_layer.bias.fill_(0.)
            self.out_layer.weight.fill_(0.)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.out_layer(x)
        return x

class DifferentialDriveWithLearnedDynamics:
    def __init__(self, learned_dyn):
        self.learned_dyn = learned_dyn

    def model(self):
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        theta = cs.MX.sym('theta')
        v = cs.MX.sym('v')
        omega = cs.MX.sym('omega')

        state = cs.vertcat(x, y, theta)
        control = cs.vertcat(v, omega)

        x_dot = v * cs.cos(theta)
        y_dot = v * cs.sin(theta)
        theta_dot = omega

        learned_residual = self.learned_dyn(state)

        f_expl = cs.vertcat(x_dot, y_dot, theta_dot) + learned_residual

        x_start = np.zeros((3,))

        model = cs.types.SimpleNamespace()
        model.x = state
        model.xdot = cs.vertcat(x_dot, y_dot, theta_dot)
        model.u = control
        model.z = cs.vertcat([])
        model.p = cs.vertcat([])
        model.f_expl = f_expl
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "ddr"

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
        t_horizon = 1.
        N = self.N
        model_ac = self.acados_model(model=model)
        model_ac.p = model.p
        nx = 3
        nu = 2
        ny = 3
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon
        if COST == 'LINEAR_LS':
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'
            ocp.cost.W = np.eye(ny)
            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:3, :3] = np.eye(3)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vz = np.array([[]])
            ocp.cost.Vx_e = np.zeros((ny, nx))
            l4c_y_expr = None
        else:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'
            x = ocp.model.x
            ocp.cost.W = np.eye(ny)
            l4c_y_expr = l4c.L4CasADi(lambda x: x[0], name='y_expr')
            ocp.model.cost_y_expr = l4c_y_expr(x)
            ocp.model.cost_y_expr_e = x[0]
        ocp.cost.W_e = np.eye(ny)
        ocp.cost.yref_e = np.zeros(3)
        ocp.cost.yref = np.zeros(3)
        ocp.constraints.x0 = model.x_start
        v_max = 1.0
        omega_max = np.pi / 4
        ocp.constraints.lbu = np.array([-v_max, -omega_max])
        ocp.constraints.ubu = np.array([v_max, omega_max])
        ocp.constraints.idxbu = np.array([0, 1])
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        if COST == 'LINEAR_LS':
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        else:
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr.name
        return ocp

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
    N = 10
    learned_dyn_model = l4c.L4CasADi(ResNet50Modified(), model_expects_batch_dim=True, name='learned_dyn')
    model = DifferentialDriveWithLearnedDynamics(learned_dyn_model)
    solver = MPC(model=model.model(), N=N, external_shared_lib_dir=learned_dyn_model.shared_lib_dir, external_shared_lib_name=learned_dyn_model.name).solver
    x = []
    x_ref = []
    ts = 1. / N
    xt = np.array([1., 1., 3.14])
    opt_times = []
    for i in range(50):
        now = time.time()
        t = np.linspace(i * ts, i * ts + 1., 10)
        yref = np.array([np.cos(0.1 * t), np.sin(0.1 * t), np.zeros_like(t)]).T
        x_ref.append(yref[0])
        for t, ref in enumerate(yref):
            solver.set(t, "yref", ref)
        solver.set(0, "lbx", xt)
        solver.set(0, "ubx", xt)
        solver.solve()
        xt = solver.get(1, "x")
        x.append(xt)
        x_l = [solver.get(i, "x") for i in range(N)]
        elapsed = time.time() - now
        opt_times.append(elapsed)

    print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')

    # Print reference and solution trajectories
    print("Reference Trajectory:")
    for ref in x_ref:
        print(ref)

    print("Solution Trajectory:")
    for sol in x:
        print(sol)

if __name__ == '__main__':
    run()
