import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from differential_drive_model import export_differential_drive_model
import scipy.linalg
from casadi import vertcat
from utils import plot_differential_drive

def differential_drive(x0, u):
    dx = u[0]*np.cos(x0[2])
    dy = u[0]*np.sin(x0[2])
    dyaw = u[1]

    return np.array([dx, dy, dyaw])

def runge_kutta(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)

    return x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)


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
    x0 = np.array([0.0, 0.0, 0.0])
    yref = np.array([1.0, 0.0, 1.57, 1.0, 0.57])
    yref_N = np.array([1.0, 0.0, 1.57])
    v_max = 1.0
    omega_max = 1.0

    Tf = 5.0
    N_horizon = 100

    ocp_solver, integrator = setup(x0, v_max, omega_max, N_horizon, Tf)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    Nsim = 100
    dt = Tf/Nsim
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0, :] = x0

    # closed loop
    for i in range(Nsim):
        # Set references for all prediction steps
        for j in range(N_horizon):
            ocp_solver.set(j, "yref", yref)
        ocp_solver.set(N_horizon, "yref", yref_N)

        # solve ocp and get next control input
        ocp_solver.set(0, "lbx", simX[i, :])
        ocp_solver.set(0, "ubx", simX[i, :])

        status = ocp_solver.solve()
        if status != 0:
            raise Exception(f'Acados returned status {status}')

        simU[i, :] = ocp_solver.get(0, "u")

        # simulate system with RK4
        simX[i + 1, :] = runge_kutta(differential_drive, simX[i, :], simU[i, :], dt)


    # plot results
    model = ocp_solver.acados_ocp.model
    plot_differential_drive(np.linspace(0, Tf, Nsim+1), v_max, omega_max, simU, simX, x_labels=model.x_labels, u_labels=model.u_labels)

if __name__ == '__main__':
    main()