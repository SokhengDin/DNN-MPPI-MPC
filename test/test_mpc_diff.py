import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from differential_drive_model import export_differential_drive_model
import scipy.linalg
from casadi import vertcat
from utils import plot_differential_drive

def setup(x0, v_max, omega_max, N_horizon, Tf):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_differential_drive_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2 * np.diag([1.0, 1.0, 0.1])
    R_mat = 2 * np.diag([0.1, 0.1])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([-v_max, -omega_max])
    ocp.constraints.ubu = np.array([v_max, omega_max])

    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

    return acados_ocp_solver, acados_integrator


def main():
    x0 = np.array([-1.0, 2.0, 3.14])
    v_max = 1.0
    omega_max = 1.0

    Tf = 5.0
    N_horizon = 100

    ocp_solver, integrator = setup(x0, v_max, omega_max, N_horizon, Tf)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    Nsim = 100
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0, :] = x0

    # closed loop
    for i in range(Nsim):
        # solve ocp and get next control input
        simU[i, :] = ocp_solver.solve_for_x0(x0_bar=simX[i, :])

        # simulate system
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=simU[i, :])

    # plot results
    model = ocp_solver.acados_ocp.model
    plot_differential_drive(np.linspace(0, Tf, Nsim+1), v_max, omega_max, simU, simX, x_labels=model.x_labels, u_labels=model.u_labels)

if __name__ == '__main__':
    main()