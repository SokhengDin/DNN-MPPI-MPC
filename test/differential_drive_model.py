from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def export_differential_drive_model() -> AcadosModel:
    model_name = 'differential_drive'

    # set up states & controls
    x = SX.sym('x')
    y = SX.sym('y')
    theta = SX.sym('theta')
    v = SX.sym('v')
    omega = SX.sym('omega')

    x = vertcat(x, y, theta)
    u = vertcat(v, omega)

    # xdot
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    theta_dot = SX.sym('theta_dot')

    xdot = vertcat(x_dot, y_dot, theta_dot)

    # dynamics
    f_expl = vertcat(
        v * cos(theta),
        v * sin(theta),
        omega
    )

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]', r'$\theta$ [rad]']
    model.u_labels = ['$v$ [m/s]', r'$\omega$ [rad/s]']
    model.t_label = '$t$ [s]'

    return model