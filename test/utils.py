import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot

def plot_differential_drive(t, v_max, omega_max, U, X_true, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
    t: time values of the discretization
    v_max: maximum linear velocity
    omega_max: maximum angular velocity
    U: array with shape (N_sim, nu)
    X_true: array with shape (N_sim+1, nx)
    latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    nx = X_true.shape[1]
    nu = U.shape[1]

    fig, axes = plt.subplots(nx+nu, 1, sharex=True)

    for i in range(nx):
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()
        if x_labels is not None:
            axes[i].set_ylabel(x_labels[i])
        else:
            axes[i].set_ylabel(f'$x_{i}$')

    for i in range(nu):
        axes[nx+i].step(t[:-1], U[:, i])
        if u_labels is not None:
            axes[nx+i].set_ylabel(u_labels[i])
        else:
            axes[nx+i].set_ylabel(f'$u_{i}$')

        if i == 0:  # Linear velocity
            axes[nx+i].hlines(v_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].hlines(-v_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].set_ylim([-1.2*v_max, 1.2*v_max])
        else:  # Angular velocity
            axes[nx+i].hlines(omega_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].hlines(-omega_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].set_ylim([-1.2*omega_max, 1.2*omega_max])

        axes[nx+i].set_xlim(t[0], t[-1])
        axes[nx+i].grid()

    axes[-1].set_xlabel(time_label)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels()

    if plt_show:
        plt.show()


def plot_race_car(t, a_max, delta_max, U, X_true, latexify=False, plt_show=True, time_label='$t$', x_labels=None, u_labels=None):
    """
    Params:
    t: time values of the discretization
    a_max: maximum acceleration
    delta_max: maximum steering angle
    U: array with shape (N_sim, nu)
    X_true: array with shape (N_sim+1, nx)
    latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    nx = X_true.shape[1]
    nu = U.shape[1]

    fig, axes = plt.subplots(nx+nu, 1, sharex=True)

    for i in range(nx):
        axes[i].plot(t, X_true[:, i])
        axes[i].grid()
        if x_labels is not None:
            axes[i].set_ylabel(x_labels[i])
        else:
            axes[i].set_ylabel(f'$x_{i}$')

    for i in range(nu):
        axes[nx+i].step(t[:-1], U[:, i])
        if u_labels is not None:
            axes[nx+i].set_ylabel(u_labels[i])
        else:
            axes[nx+i].set_ylabel(f'$u_{i}$')

        if i == 0:  # Acceleration
            axes[nx+i].hlines(a_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].hlines(-a_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].set_ylim([-1.2*a_max, 1.2*a_max])
        else:  # Steering angle
            axes[nx+i].hlines(delta_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].hlines(-delta_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
            axes[nx+i].set_ylim([-1.2*delta_max, 1.2*delta_max])

        axes[nx+i].set_xlim(t[0], t[-1])
        axes[nx+i].grid()

    axes[-1].set_xlabel(time_label)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)
    fig.align_ylabels()

    if plt_show:
        plt.show()