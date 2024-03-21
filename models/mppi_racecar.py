import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import patches
from matplotlib.animation import ArtistAnimation
from IPython import display



class Vehicle:

    def __init__(
            self,
            wheel_base: float = 2.5, #[m],
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[-100.0, 0.0], [100.0, 0.0]]),
            delta_t: float = 0.05, # [s]
            visualize: bool = True
    ) -> None:
        """ Intialize vehicle in the racecar environment
            state variables:
                x: x-axis position in the global frame [m]
                y: y-axis position in the global frame [m]
                yaw: orientation in the global frame [rad]
                v: longitudinal velocity [m/s]
            control input:
                steer: front tire angle of the vehicle [rad] (positive in the counterclockwize direction)
                accel: longitudinal acceleration of the vehicle [m/s^2] (positive in the forward direction)
        """
        # vehicle parameters
        self.wheel_base = wheel_base #[m]
        self.max_steer_abs = max_steer_abs #[rad]
        self.max_accel_abs = max_accel_abs #[m/s^2]
        self.delta_t = delta_t # [s]
        self.ref_path = ref_path

        # visualization settings
        self.vehicle_w = 3.00
        self.vehicle_l = 4.00
        self.view_x_lim_min, self.view_x_lim_max = -20.0, 20.0
        self.view_y_lim_min, self.view_y_lim_max = -25.0, 25.0

        # reset environment
        self.visualize_flag = visualize
        self.reset()


    def reset(
            self,
            init_state: np.ndarray = np.array([0.0, 0.0, 0.0, 0.0]), # [x, y, yaw, v]
        ) -> None:
        """Reset environment to initial state"""

        # reset state variables
        self.state = init_state

        # clear animation frames
        self.frames = []

        if self.visualize_flag:
            # prepare figure
            self.fig = plt.figure(figsize=(9, 9))
            self.main_ax = plt.subplot2grid((3, 4), (0, 0), rowspan=3, colspan=3)
            self.minimap_ax = plt.subplot2grid((3, 4), (0, 3))
            self.steer_ax = plt.subplot2grid((3, 4), (1, 3))
            self.accel_ax = plt.subplot2grid((3, 4), (2, 3))

             # graph layout settings
            ## main view
            self.main_ax.set_aspect('equal')
            self.main_ax.set_xlim(self.view_x_lim_min, self.view_x_lim_max)
            self.main_ax.set_ylim(self.view_y_lim_min, self.view_y_lim_max)
            self.main_ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            self.main_ax.tick_params(bottom=False, left=False, right=False, top=False)
            ## mini map
            self.minimap_ax.set_aspect('equal')
            self.minimap_ax.axis('off')
            ## steering angle view
            self.steer_ax.set_title("Steering Angle", fontsize=12)
            self.steer_ax.axis('off')
            ## acceleration view
            self.accel_ax.set_title("Acceleration", fontsize=12)
            self.accel_ax.axis('off')

            # apply tight layout
            self.fig.tight_layout()

    def update(
            self,
            u: np.ndarray,
            delta_t: float = 0.0,
            append_frame: bool = True,
            optimal_traj: np.ndarray = np.empty(0), # predicted optimal trajectory from mppi
            sampled_traj_list: np.ndarray = np.empty(0), # sampled trajectories from mppi
        ) -> None:
        """Update vehicle state and visualize the vehicle"""
        # keep previous states
        x, y, yaw, v = self.state

        # prepare params
        l = self.wheel_base
        dt = self.delta_t if delta_t == 0.0 else delta_t

        # limit control inputs
        steer = np.clip(u[0], -self.max_steer_abs, self.max_steer_abs)
        accel = np.clip(u[1], -self.max_accel_abs, self.max_accel_abs)

        # update state variables
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt
        self.state = np.array([new_x, new_y, new_yaw, new_v])

        # record frame
        if append_frame:
            self.append_frame(steer, accel, optimal_traj, sampled_traj_list)
        
    
    def get_state(self) -> np.ndarray:
        """Return current state"""
        return self.state
    
    def append_frame(
            self,
            steer: float,
            accel: float,
            optimal_traj: np.ndarray,
            sampled_traj_list: np.ndarray   
        ) -> None:
        # Get current state
        x, y, yaw, v = self.state

        ### main view ###
        # draw the vehicle shape
        vw, vl = self.vehicle_w, self.vehicle_l
        vehicle_shape_x = [-0.5*vl, -0.5*vl, +0.5*vl, +0.5*vl, -0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, -0.5*vw, -0.5*vw, 0.0]
        rotated_vehicle_shape_x, rotated_vehicle_shape_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [0, 0]) # make the vehicle be at the center of the figure
        frame = self.main_ax.plot(rotated_vehicle_shape_x, rotated_vehicle_shape_y, color='black', linewidth=2.0, zorder=3)

        # draw wheels
        ww, wl = 0.4, 0.7 #[m]
        wheel_shape_x = np.array([-0.5*wl, -0.5*wl, +0.5*wl, +0.5*wl, -0.5*wl, -0.5*wl])
        wheel_shape_y = np.array([0.0, +0.5*ww, +0.5*ww, -0.5*ww, -0.5*ww, 0.0])

        ## rear-left wheel
        wheel_shape_rl_x, wheel_shape_rl_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, 0.3*vw])
        wheel_rl_x, wheel_rl_y = \
            self._affine_transform(wheel_shape_rl_x, wheel_shape_rl_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_rl_x, wheel_rl_y, color='black', zorder=3)

        ## rear-right wheel
        wheel_shape_rr_x, wheel_shape_rr_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, 0.0, [-0.3*vl, -0.3*vw])
        wheel_rr_x, wheel_rr_y = \
            self._affine_transform(wheel_shape_rr_x, wheel_shape_rr_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_rr_x, wheel_rr_y, color='black', zorder=3)

        ## front-left wheel
        wheel_shape_fl_x, wheel_shape_fl_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, steer, [0.3*vl, 0.3*vw])
        wheel_fl_x, wheel_fl_y = \
            self._affine_transform(wheel_shape_fl_x, wheel_shape_fl_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_fl_x, wheel_fl_y, color='black', zorder=3)

        ## front-right wheel
        wheel_shape_fr_x, wheel_shape_fr_y = \
            self._affine_transform(wheel_shape_x, wheel_shape_y, steer, [0.3*vl, -0.3*vw])
        wheel_fr_x, wheel_fr_y = \
            self._affine_transform(wheel_shape_fr_x, wheel_shape_fr_y, yaw, [0, 0])
        frame += self.main_ax.fill(wheel_fr_x, wheel_fr_y, color='black', zorder=3)


        # draw the vehicle center circle
        vehicle_center = patches.Circle([0, 0], radius=vw/20.0, fc='white', ec='black', linewidth=2.0, zorder=6)
        frame += [self.main_ax.add_artist(vehicle_center)]

        # draw the reference path
        ref_path_x = self.ref_path[:, 0] - np.full(self.ref_path.shape[0], x)
        ref_path_y = self.ref_path[:, 1] - np.full(self.ref_path.shape[0], y)
        frame += self.main_ax.plot(ref_path_x, ref_path_y, color='black', linestyle="dashed", linewidth=1.5)

        # draw the information text
        text = "vehicle velocity = {v:>+6.1f} [m/s]".format(pos_e=x, head_e=np.rad2deg(yaw), v=v)
        frame += [self.main_ax.text(0.5, 0.02, text, ha='center', transform=self.main_ax.transAxes, fontsize=14, fontfamily='monospace')]

        # draw the predicted optimal trajectory from mppi
        if optimal_traj.any():
            optimal_traj_x_offset = np.ravel(optimal_traj[:, 0]) - np.full(optimal_traj.shape[0], x)
            optimal_traj_y_offset = np.ravel(optimal_traj[:, 1]) - np.full(optimal_traj.shape[0], y)
            frame += self.main_ax.plot(optimal_traj_x_offset, optimal_traj_y_offset, color='#990099', linestyle="solid", linewidth=1.5, zorder=5)

        # draw the sampled trajectories from mppi
        if sampled_traj_list.any():
            min_alpha_value = 0.25
            max_alpha_value = 0.35
            for idx, sampled_traj in enumerate(sampled_traj_list):
                # draw darker for better samples
                alpha_value = (1.0 - (idx+1)/len(sampled_traj_list)) * (max_alpha_value - min_alpha_value) + min_alpha_value
                sampled_traj_x_offset = np.ravel(sampled_traj[:, 0]) - np.full(sampled_traj.shape[0], x)
                sampled_traj_y_offset = np.ravel(sampled_traj[:, 1]) - np.full(sampled_traj.shape[0], y)
                frame += self.main_ax.plot(sampled_traj_x_offset, sampled_traj_y_offset, color='gray', linestyle="solid", linewidth=0.2, zorder=4, alpha=alpha_value)

        ### mini map view ###
        frame += self.minimap_ax.plot(self.ref_path[:, 0], self.ref_path[:,1], color='black', linestyle='dashed')
        rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [x, y]) # make the vehicle be at the center of the figure
        frame += self.minimap_ax.plot(rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap, color='black', linewidth=2.0, zorder=3)
        frame += self.minimap_ax.fill(rotated_vehicle_shape_x_minimap, rotated_vehicle_shape_y_minimap, color='white', zorder=2)

        ### control input view ###
        # steering angle
        MAX_STEER = self.max_steer_abs
        PIE_RATE = 3.0/4.0
        PIE_STARTANGLE = 225 # [deg]
        s_abs = np.abs(steer)
        if steer < 0.0: # when turning right
            steer_pie_obj, _ = self.steer_ax.pie([MAX_STEER*PIE_RATE, s_abs*PIE_RATE, (MAX_STEER-s_abs)*PIE_RATE, 2*MAX_STEER*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        else: # when turning left
            steer_pie_obj, _ = self.steer_ax.pie([(MAX_STEER-s_abs)*PIE_RATE, s_abs*PIE_RATE, MAX_STEER*PIE_RATE, 2*MAX_STEER*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        frame += steer_pie_obj
        frame += [self.steer_ax.text(0, -1, f"{np.rad2deg(steer):+.2f} " + r"$ \rm{[deg]}$", size = 14, horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # acceleration
        MAX_ACCEL = self.max_accel_abs
        PIE_RATE = 3.0/4.0
        PIE_STARTANGLE = 225 # [deg]
        a_abs = np.abs(accel)
        if accel > 0.0:
            accel_pie_obj, _ = self.accel_ax.pie([MAX_ACCEL*PIE_RATE, a_abs*PIE_RATE, (MAX_ACCEL-a_abs)*PIE_RATE, 2*MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        else:
            accel_pie_obj, _ = self.accel_ax.pie([(MAX_ACCEL-a_abs)*PIE_RATE, a_abs*PIE_RATE, MAX_ACCEL*PIE_RATE, 2*MAX_ACCEL*(1-PIE_RATE)], startangle=PIE_STARTANGLE, counterclock=False, colors=["lightgray", "black", "lightgray", "white"], wedgeprops={'linewidth': 0, "edgecolor":"white", "width":0.4})
        frame += accel_pie_obj
        frame += [self.accel_ax.text(0, -1, f"{accel:+.2f} " + r"$ \rm{[m/s^2]}$", size = 14, horizontalalignment='center', verticalalignment='center', fontfamily='monospace')]

        # append frame
        self.frames.append(frame)
    
    # rotate shape and return location on the x-y plane.
    def _affine_transform(self, xlist: list, ylist: list, angle: float, translation: list=[0.0, 0.0]) -> Tuple[list, list]:
        transformed_x = []
        transformed_y = []
        if len(xlist) != len(ylist):
            print("[ERROR] xlist and ylist must have the same size.")
            raise AttributeError

        for i, xval in enumerate(xlist):
            transformed_x.append((xlist[i])*np.cos(angle)-(ylist[i])*np.sin(angle)+translation[0])
            transformed_y.append((xlist[i])*np.sin(angle)+(ylist[i])*np.cos(angle)+translation[1])
        transformed_x.append(transformed_x[0])
        transformed_y.append(transformed_y[0])
        return transformed_x, transformed_y
    
    def show_animation(self, interval_ms: int) -> None:
        """show animation of the recorded frames"""
        ani = ArtistAnimation(self.fig, self.frames, interval=interval_ms) # blit=True
        html = display.HTML(ani.to_jshtml())
        display.display(html)
        plt.close()

    def save_animation(self, filename: str, interval: int, movie_writer: str="ffmpeg") -> None:
        """save animation of the recorded frames (ffmpeg required)"""
        ani = ArtistAnimation(self.fig, self.frames, interval=interval)
        ani.save(filename, writer=movie_writer)


class MPPIControllerPathTracking:

    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, #[m],
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            num_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]),
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]),
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]),
            visualize_optimal_traj = True,
            visualze_sampled_trajs = False
        ) -> None:
            """Initialize MPPI Controller"""
            # mppi parameters
            self.dim_x = 4
            self.dim_u = 2
            self.T = horizon_step_T
            self.K = num_samples_K
            self.param_exploration = param_exploration
            self.param_lambda = param_lambda
            self.param_alpha = param_alpha
            self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
            self.Sigma = sigma
            self.stage_cost_weight = stage_cost_weight
            self.terminal_cost_weight = terminal_cost_weight
            self.visualize_optimal_traj = visualize_optimal_traj
            self.visualze_sampled_trajs = visualze_sampled_trajs

            # vehicle parameters
            self.delta_t = delta_t  #[s]
            self.wheel_base = wheel_base
            self.max_steer_abs = max_steer_abs
            self.max_accel_abs = max_accel_abs
            self.ref_path = ref_path   

            # mppi variables
            self.u_prev = np.zeros((self.T, self.K))

            # ref_path info
            self.prev_waypoints_idx = 0

    def calc_control_input(
            self,
            observed_x: np.ndarray   
        ) -> Tuple[float, np.ndarray]:
        """Calculate control input using MPPI"""
        # load previous control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # get the waypoint closed to current vehicle position
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError
        
        # prepare buffer
        S = np.zeros((self.K))

        # sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # prepare buffer of sampled control input sequence
        v = np.zeros((self.K, self.T, self.dim_u))

        # loop for 0 ~ K-1 samples
        for k in range(self.K):
            # set initial(t=0) observed state of the vehicle
            x = x0

            # loop for time step t = 1 ~ T
            for t in range(1, self.T+1):

                # get contorl input with noise
                if k < (1.0-self.param_exploration)*self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1] # sampling for exploitation
                else:
                    v[k, t-1] = epsilon[k, t-1] # sampling for exploration

                # update x
                x = self._F(x, self._g(v[k, t-1]))

                # add stage cost
                S[k] += self._c(x) + self.param_gamma*u[t-1].T@np.linalg.inv(self.Sigma)@v[k, t-1]

            # add terminal cost
            S[k] += self._phi(x)
        
        # Compute weight
        w = self._compute_weights(S)

        # calculate w_k * epsilon_k
        w_epsilon = np.zeros((self.T, self.dim_u))
        for t in range(self.T): # loop for time step t = 0 ~ T-1
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x = x0
            for t in range(self.T):
                x = self._F(x, self._g(u[t-1]))
                optimal_traj[t] = x
        
        # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        sorted_idx = np.argsort(S) # sort samples by state cost, 0th is the best sample
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._F(x, self._g(v[k, t-1]))
                    sampled_traj_list[k, t] = x

        # update previous control input sequence
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        return u[0], optimal_traj, sampled_traj_list

        
    def _calc_epsilon(
            self,
            sigma: np.ndarray,
            size_sample: int,
            size_time_step: int,
            size_dim_u: np.ndarray
        )->  np.ndarray:
        """Calculate sample epsilon """
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError
        
        # Sample epsilon
        mu = np.zeros((size_dim_u))
        epsilon = np.random.multivariate_normal(
            mu,
            sigma,
            (size_sample, size_time_step)
        )
        return epsilon
    
    def _g(self, v: np.ndarray) -> float:
        """Clamp input"""
        # limit input inputs
        v[0] = np.clip(v[0], -self.max_steer_abs, self.max_steer_abs)
        v[1] = np.clip(v[1], -self.max_accel_abs, self.max_accel_abs)
        return v
    
    def _c(self, x_t: np.ndarray) -> float:
        # parse x_t
        x, y, yaw, v = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))

        # Calculate stage cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                     self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
        return stage_cost
    
    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, y, yaw, v = x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate terminal cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
        return terminal_cost
    
    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v
    
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        x, y, yaw, v = x_t
        steer, accel = v_t

        # prepare params
        l = self.wheel_base
        dt = self.delta_t

        # update state variables
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt

        # return updated state
        x_t_plus_1 = np.array([new_x, new_y, new_yaw, new_v])
        return x_t_plus_1
    
    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )

        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w
    
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average filter for smoothing input sequence"""
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:, d] = np.convolve(xx[:, d], b, mode='same')
            n_conv = math.ceil(window_size/2)
            xx_mean[0, d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i, d] *= window_size/(n_conv+i)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2))
        return xx_mean