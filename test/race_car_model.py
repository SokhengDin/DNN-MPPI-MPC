from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, tan

def export_race_car_model() -> AcadosModel:
    model_name = 'race_car'

    # set up states & controls
    x = SX.sym('x')
    y = SX.sym('y')
    yaw = SX.sym('yaw')
    v = SX.sym('v')
    a = SX.sym('a')
    delta = SX.sym('delta')

    x = vertcat(x, y, yaw, v)
    u = vertcat(a, delta)

    # xdot
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    yaw_dot = SX.sym('yaw_dot')
    v_dot = SX.sym('v_dot')

    xdot = vertcat(x_dot, y_dot, yaw_dot, v_dot)

    # dynamics
    L = 2.5  # wheelbase length [m]
    f_expl = vertcat(
        v * cos(yaw),
        v * sin(yaw),
        v * tan(delta) / L,
        a
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
    model.x_labels = ['$x$ [m]', '$y$ [m]', r'$\psi$ [rad]', '$v$ [m/s]']
    model.u_labels = ['$a$ [m/s^2]', r'$\delta$ [rad]']
    model.t_label = '$t$ [s]'

    return model