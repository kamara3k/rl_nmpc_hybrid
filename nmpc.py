import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time


class ReactorModel:
    """
    Reactor model aligned with envs.py.

    Plant is always the true 8-drum HolosPK-like reactor.
    Control interface:
      - num_drums=1  -> single shared increment replicated to all 8 drums
                        (matches HolosSingle.step in envs.py)
      - num_drums=8  -> 8 independent drum increments
                        (matches HolosMulti.step in envs.py)

    State ordering kept the same as the original NMPC file:
      x = [n_r, c1..c6, Xe, I, Tf, Tm, Tc]
    """
    def __init__(self, num_drums=1, dt=1, Ts=None):
        self.dt = dt
        self.Ts = dt if Ts is None else Ts
        self.num_drums = int(num_drums)
        if self.num_drums not in (1, 8):
            raise ValueError("num_drums must be 1 or 8 to align with envs.py")

        # True plant has 8 physical drums as in envs.py / HolosPK
        self.n_physical_drums = 8

        # Neutronic / xenon-iodine data
        self.sigma_Xe = 2.65e-22
        self.yield_I = 0.061
        self.yield_Xe = 0.002
        self.lambda_Xe = 2.09e-5
        self.lambda_I = 2.87e-5
        self.Sigma_f = 0.1117
        self.therm_n_vel = 2.19e3
        self.neutron_lifetime = 1.68e-3

        # Delayed neutron data
        self.beta = 0.004801
        self.betas = np.array([
            1.42481E-04,
            9.24281E-04,
            7.79956E-04,
            2.06583E-03,
            6.71175E-04,
            2.17806E-04,
        ])
        self.lambdas = np.array([
            1.272E-02,
            3.174E-02,
            1.160E-01,
            3.110E-01,
            1.400E+00,
            3.870E+00,
        ])

        # Thermal-hydraulic data
        self.cp_f = 977
        self.cp_m = 1697
        self.cp_c = 5188.6
        self.M_f = 2002
        self.M_m = 11573
        self.M_c = 500
        self.heat_f = 0.96
        self.Tf0 = 832.4
        self.Tm0 = 830.22
        self.T_in = 795.47
        self.T_out = 1106
        self.Tc0 = 814.35
        self.K_fm = 1.17e6
        self.K_mc = 2.16e5
        self.M_dot = 17.5

        # Reactivity coefficients
        self.alpha_f = -2.875e-5
        self.alpha_m = -3.696e-5
        self.alpha_c = 0.0

        # Neutron density / power scaling
        self.n_0 = 2.25e13
        self.P_r = 22e6

        # Drum parameters (per drum, matching envs.py)
        self.u0 = 77.8
        self.rho_max = 0.00510
        self.rho_ss = self.rho_max * (1 - np.cos(np.deg2rad(self.u0))) / 2.0
        assert self.rho_ss < self.rho_max

        # Steady-state Xe & I
        self.I0 = self.yield_I * self.Sigma_f * self.therm_n_vel * self.n_0 / self.lambda_I
        self.Xe0 = (
            self.yield_Xe * self.Sigma_f * self.therm_n_vel * self.n_0
            + self.lambda_I * self.I0
        ) / (self.lambda_Xe + self.sigma_Xe * self.therm_n_vel * self.n_0)

    @property
    def control_dim(self):
        return 1 if self.num_drums == 1 else 8

    def get_initial_state(self, initial_power: float = 1.0) -> np.ndarray:
        n_r = float(initial_power)
        c1 = c2 = c3 = c4 = c5 = c6 = n_r
        Tf = self.Tf0
        Tm = self.Tm0
        Tc = self.Tc0
        Xe = self.Xe0
        I = self.I0
        return np.array([n_r, c1, c2, c3, c4, c5, c6, Xe, I, Tf, Tm, Tc], dtype=float)

    def get_initial_mv(self) -> np.ndarray:
        if self.num_drums == 1:
            return np.array([self.u0], dtype=float)
        return np.full(8, self.u0, dtype=float)

    def expand_control(self, u):
        """Map controller MV to the true 8 physical drum angles."""
        u = np.asarray(u, dtype=float).reshape(-1)
        if self.num_drums == 1:
            if u.size != 1:
                raise ValueError(f"Expected scalar/shared control, got shape {u.shape}")
            return np.full(self.n_physical_drums, float(u[0]), dtype=float)
        if u.size != self.n_physical_drums:
            raise ValueError(f"Expected 8 drum angles, got shape {u.shape}")
        return u.copy()

    def clip_mv(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)
        if self.num_drums == 1:
            return np.array([float(np.clip(u[0], 0.0, 180.0))], dtype=float)
        return np.clip(u, 0.0, 180.0)

    def _drum_reactivity(self, drum_angles_deg) -> float:
        """Exact 8-drum reactivity from envs.py / HolosPK.calc_reactivity."""
        drum_angles_deg = np.asarray(drum_angles_deg, dtype=float).reshape(self.n_physical_drums)
        return float(np.sum(self.rho_max * (1 - np.cos(np.deg2rad(drum_angles_deg))) / 2.0 - self.rho_ss))

    def continuous_dynamics(self, x: np.ndarray, drum_angles_deg) -> np.ndarray:
        n_r, c1, c2, c3, c4, c5, c6, Xe, I, Tf, Tm, Tc = x

        rho_drum = self._drum_reactivity(drum_angles_deg)
        rho = (
            rho_drum
            + self.alpha_f * (Tf - self.Tf0)
            + self.alpha_m * (Tm - self.Tm0)
            - self.sigma_Xe * (Xe - self.Xe0) / self.Sigma_f
        )

        # Numerical guard: keep prediction finite inside optimizer.
        rho = float(np.clip(rho, -0.05, self.beta - 1e-6))

        precursor_concs = np.array([c1, c2, c3, c4, c5, c6])
        d_n_r = (((rho - self.beta) * n_r) + np.sum(self.betas * precursor_concs)) / self.neutron_lifetime
        d_c = self.lambdas * n_r - self.lambdas * precursor_concs

        d_Tf = (self.heat_f * self.P_r * n_r - self.K_fm * (Tf - Tc)) / (self.M_f * self.cp_f)
        d_Tm = (((1.0 - self.heat_f) * self.P_r * n_r) + self.K_fm * (Tf - Tm) - self.K_mc * (Tm - Tc)) / (self.M_m * self.cp_m)
        d_Tc = (self.K_mc * (Tm - Tc) - 2.0 * self.M_dot * self.cp_c * (Tc - self.T_in)) / (self.M_c * self.cp_c)

        n_rate_density = self.therm_n_vel * self.n_0 * n_r
        d_I = self.yield_I * self.Sigma_f * n_rate_density - self.lambda_I * I
        d_Xe = (
            self.yield_Xe * self.Sigma_f * n_rate_density
            + self.lambda_I * I
            - self.lambda_Xe * Xe
            - self.sigma_Xe * Xe * n_rate_density
        )

        dx = np.zeros_like(x, dtype=float)
        dx[0] = d_n_r
        dx[1:7] = d_c
        dx[7] = d_Xe
        dx[8] = d_I
        dx[9] = d_Tf
        dx[10] = d_Tm
        dx[11] = d_Tc
        return dx

    def discrete_dynamics(self, x: np.ndarray, u, Ts: float = None) -> np.ndarray:
        """Heun integrator, using true 8-drum physical angles."""
        if Ts is None:
            Ts = self.Ts
        drum_angles = self.expand_control(u)
        M = 5
        delta = Ts / M
        xk1 = np.asarray(x, dtype=float).copy()
        for _ in range(M):
            f1 = self.continuous_dynamics(xk1, drum_angles)
            hx = xk1 + delta * f1
            f2 = self.continuous_dynamics(hx, drum_angles)
            xk1 = xk1 + delta * (f1 + f2) / 2.0
            if not np.all(np.isfinite(xk1)):
                return np.full_like(xk1, np.nan)
        return xk1

    def output_function(self, x: np.ndarray) -> float:
        return float(x[0])


class ExtendedKalmanFilter:
    def __init__(self, reactor_model, nx=12):
        self.reactor_model = reactor_model
        self.nx = nx
        self.P = np.eye(nx)
        self.Q = 0.01 * np.eye(nx)
        self.R = 0.1
        self.H = np.array([[1] + [0] * (nx - 1)])

    def numerical_jacobian(self, x, u, Ts):
        epsilon = 1e-5
        J = np.zeros((self.nx, self.nx))
        fx = self.reactor_model.discrete_dynamics(x, u, Ts)
        for i in range(self.nx):
            x_eps = x.copy()
            x_eps[i] += epsilon
            fx_eps = self.reactor_model.discrete_dynamics(x_eps, u, Ts)
            J[:, i] = (fx_eps - fx) / epsilon
        return J

    def update(self, x, y, u, Ts):
        xk_pred = self.reactor_model.discrete_dynamics(x, u, Ts)
        F = self.numerical_jacobian(x, u, Ts)
        P_pred = F @ self.P @ F.T + self.Q
        S = (self.H @ P_pred @ self.H.T + self.R).item()
        K = (P_pred @ self.H.T) / S
        innovation = float(y) - (self.H @ xk_pred).item()
        xk = xk_pred + (K.flatten() * innovation)
        self.P = (np.eye(self.nx) - K @ self.H) @ P_pred
        return xk


class NonlinearMPC:
    def __init__(self, reactor_model, prediction_horizon=10, control_horizon=3):
        self.value_fn = None
        self.use_terminal_value = True
        self.terminal_weight = 1.0
        self.enable_value_when_td_below = 0.05
        self._value_td_uncertainty = float('inf')

        self.reactor_model = reactor_model
        self.prediction_horizon = int(prediction_horizon)
        self.control_horizon = int(control_horizon)

        self.Q = 1e7
        self.R = 0.02
        self.Rd = 0.02

        self.u_lb = 0.0
        self.u_ub = 180.0
        self.du_lb = -1.0
        self.du_ub = 1.0

        self.penalty_bounds = 1e8
        self.penalty_rate = 1e8
        self.maxiter = 40
        self._last_solution = None

    def _as_mv(self, mv):
        mv = np.asarray(mv, dtype=float).reshape(-1)
        if self.reactor_model.num_drums == 1:
            if mv.size == 1:
                return mv.copy()
            return np.array([float(np.mean(mv))], dtype=float)
        if mv.size == 1:
            return np.full(8, float(mv[0]), dtype=float)
        return mv.copy()

    def _control_shape(self):
        return self.reactor_model.control_dim

    def _rollout_from_increment_sequence(self, xk, prev_mv, du_flat, ref):
        Nc = self.control_horizon
        Np = self.prediction_horizon
        nu = self._control_shape()
        du_seq = np.asarray(du_flat, dtype=float).reshape(Nc, nu)
        prev_mv = self._as_mv(prev_mv)
        x = np.asarray(xk, dtype=float).copy()
        mv = prev_mv.copy()
        J = 0.0
        last_mv = prev_mv.copy()

        for k in range(Np):
            idx = min(k, Nc - 1)
            mv = self.reactor_model.clip_mv(mv + du_seq[idx])
            x = self.reactor_model.discrete_dynamics(x, mv)
            if not np.all(np.isfinite(x)):
                return 1e12
            y = self.reactor_model.output_function(x)
            err = float(y) - float(ref)
            J += self.Q * (err ** 2)
            if k < Nc:
                J += self.R * float(np.sum(mv ** 2))
                J += self.Rd * float(np.sum((mv - last_mv) ** 2))
                last_mv = mv.copy()

        if (self.value_fn is not None) and bool(getattr(self, 'use_terminal_value', False)):
            try:
                J += float(self.terminal_weight) * float(self.value_fn(x))
            except Exception:
                pass
        return float(J) if np.isfinite(J) else 1e12

    def _penalized_objective(self, du_flat, xk, prev_mv, ref):
        Nc = self.control_horizon
        nu = self._control_shape()
        du_seq = np.asarray(du_flat, dtype=float).reshape(Nc, nu)
        prev_mv = self._as_mv(prev_mv)
        mv = prev_mv.copy()
        Jp = 0.0

        du_lb_adj = self.du_lb * self.reactor_model.dt
        du_ub_adj = self.du_ub * self.reactor_model.dt

        for k in range(Nc):
            du = du_seq[k]
            # rate penalty on increments directly
            low_violation = np.maximum(du_lb_adj - du, 0.0)
            high_violation = np.maximum(du - du_ub_adj, 0.0)
            Jp += self.penalty_rate * float(np.sum(low_violation ** 2 + high_violation ** 2))

            mv = mv + du
            lb_violation = np.maximum(self.u_lb - mv, 0.0)
            ub_violation = np.maximum(mv - self.u_ub, 0.0)
            Jp += self.penalty_bounds * float(np.sum(lb_violation ** 2 + ub_violation ** 2))

        J = self._rollout_from_increment_sequence(xk, prev_mv, du_flat, ref)
        J_total = J + Jp
        return float(J_total) if np.isfinite(J_total) else 1e12

    def _initial_guess(self):
        Nc = self.control_horizon
        nu = self._control_shape()
        if self._last_solution is not None and self._last_solution.size == Nc * nu:
            old = self._last_solution.reshape(Nc, nu)
            shifted = np.vstack([old[1:], old[-1:]])
            return shifted.reshape(-1)
        return np.zeros(Nc * nu, dtype=float)

    def calculate_control(self, xk, prev_mv, ref):
        try:
            self.use_terminal_value = (self._value_td_uncertainty <= self.enable_value_when_td_below)
        except Exception:
            pass

        nu = self._control_shape()
        Nc = self.control_horizon
        du_init = self._initial_guess()
        du_bound = max(abs(self.du_lb), abs(self.du_ub)) * self.reactor_model.dt
        bounds = [(-du_bound, du_bound)] * (Nc * nu)

        try:
            res = minimize(
                self._penalized_objective,
                du_init,
                args=(xk, prev_mv, ref),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': self.maxiter, 'ftol': 1e-6, 'maxls': 20},
            )
            du_star = res.x if (res is not None and np.all(np.isfinite(res.x))) else du_init
        except Exception:
            du_star = du_init

        self._last_solution = np.asarray(du_star, dtype=float).copy()
        first_du = np.asarray(du_star, dtype=float).reshape(Nc, nu)[0]
        prev_mv = self._as_mv(prev_mv)
        mv = self.reactor_model.clip_mv(prev_mv + first_du)
        return mv


# Kamal_start: MPC-as-critic helper adapted for vector/shared MV

def evaluate_q_fixed_first(
    self,
    xk: np.ndarray,
    prev_mv,
    ref: float,
    u0_fixed,
    prediction_horizon: int | None = None,
    control_horizon: int | None = None,
    optimize_tail: bool = True,
) -> float:
    model = getattr(self, 'reactor_model', None) or getattr(self, 'reactor', None) or getattr(self, 'model', None)
    if model is None:
        raise AttributeError('NonlinearMPC.evaluate_q_fixed_first: missing model.')

    Np = int(prediction_horizon) if prediction_horizon is not None else int(getattr(self, 'prediction_horizon', 10))
    Nc = int(control_horizon) if control_horizon is not None else int(getattr(self, 'control_horizon', 3))
    Np = max(1, Np)
    Nc = max(1, Nc)

    prev_mv = self._as_mv(prev_mv)
    u0_fixed = self._as_mv(u0_fixed)

    if (Nc == 1) or (not optimize_tail):
        x = np.asarray(xk, dtype=float).copy()
        mv = u0_fixed.copy()
        J = 0.0
        last_mv = prev_mv.copy()
        for k in range(Np):
            x = model.discrete_dynamics(x, mv)
            if not np.all(np.isfinite(x)):
                return float(1e12)
            y = model.output_function(x)
            J += float(getattr(self, 'Q', 1.0)) * (float(y) - float(ref)) ** 2
            if k < Nc:
                J += float(getattr(self, 'R', 0.0)) * float(np.sum(mv ** 2))
                J += float(getattr(self, 'Rd', 0.0)) * float(np.sum((mv - last_mv) ** 2))
                last_mv = mv.copy()
        return float(J) if np.isfinite(J) else float(1e12)

    nu = model.control_dim
    tail_dim = (Nc - 1) * nu
    u_init = np.tile(u0_fixed, Nc - 1)
    bounds = [(float(getattr(self, 'u_lb', 0.0)), float(getattr(self, 'u_ub', 180.0)))] * tail_dim

    def tail_cost(u_tail):
        try:
            u_tail = np.asarray(u_tail, dtype=float).reshape(Nc - 1, nu)
            u_full = np.vstack([u0_fixed.reshape(1, nu), u_tail])
            x = np.asarray(xk, dtype=float).copy()
            J = 0.0
            last_mv = prev_mv.copy()
            for k in range(Np):
                mv = u_full[min(k, Nc - 1)]
                x = model.discrete_dynamics(x, mv)
                if not np.all(np.isfinite(x)):
                    return 1e12
                y = model.output_function(x)
                J += float(getattr(self, 'Q', 1.0)) * (float(y) - float(ref)) ** 2
                if k < Nc:
                    J += float(getattr(self, 'R', 0.0)) * float(np.sum(mv ** 2))
                    J += float(getattr(self, 'Rd', 0.0)) * float(np.sum((mv - last_mv) ** 2))
                    last_mv = mv.copy()
            return float(J) if np.isfinite(J) else 1e12
        except Exception:
            return 1e12

    try:
        res = minimize(tail_cost, u_init, method='L-BFGS-B', bounds=bounds, options={'maxiter': 30, 'ftol': 1e-6})
        if res is not None and np.isfinite(res.fun):
            return float(res.fun)
    except Exception:
        pass
    return float(tail_cost(u_init))

# Kamal_end: MPC-as-critic helper adapted for vector/shared MV


ref_fun = interp1d(
    [0, 25, 35, 50, 60, 80, 125, 135, 150, 175, 200],
    np.array([100, 100, 80, 70, 70, 50, 50, 60, 60, 90, 90]) / 100.0,
    bounds_error=False,
    fill_value=(1.0, 0.9),
)


class Simulator:
    def __init__(self, reactor_model, ekf, controller, duration=2000, ref_fun_override=None, reference=None):
        self.reactor_model = reactor_model
        self.ekf = ekf
        self.controller = controller
        self.duration = duration

        self.dt = reactor_model.dt
        self.sim_time = np.arange(0, duration + self.dt, self.dt)
        print('len(sim_time): ', len(self.sim_time))

        if reference is not None:
            reference = np.asarray(reference, dtype=float).reshape(-1)
            if reference.shape[0] != self.sim_time.shape[0]:
                raise ValueError(f'reference length mismatch: got {reference.shape[0]}, expected {self.sim_time.shape[0]}')
            self.reference = reference
        else:
            ref_callable = ref_fun_override if ref_fun_override is not None else ref_fun
            self.reference = np.asarray(ref_callable(self.sim_time), dtype=float).reshape(-1)

        print('len(reference): ', len(self.reference))
        self.nt = len(self.sim_time)
        self.nx = 12
        self.x_history = np.zeros((self.nx, self.nt))
        self.xk_history = np.zeros((self.nx, self.nt))
        mv_dim = self.reactor_model.control_dim
        self.mv_history = np.zeros((mv_dim, self.nt))

    def run_simulation(self):
        x = self.reactor_model.get_initial_state(self.reference[0])
        mv = self.reactor_model.get_initial_mv()
        y = x[0]

        self.x_history[:, 0] = x
        self.xk_history[:, 0] = x
        self.mv_history[:, 0] = mv

        print('Starting simulation...')
        start_time = time.time()

        for ct in range(1, self.nt):
            xk = self.ekf.update(self.xk_history[:, ct - 1], y, mv, self.reactor_model.Ts)
            mv = self.controller.calculate_control(xk, self.mv_history[:, ct - 1], self.reference[ct])
            x = self.reactor_model.discrete_dynamics(x, mv)
            y = self.reactor_model.output_function(x)

            self.x_history[:, ct] = x
            self.xk_history[:, ct] = xk
            self.mv_history[:, ct] = mv

            if ct % 50 == 0:
                print(f'Simulation progress: {ct}/{self.nt} steps')

        end_time = time.time()
        print(f'Simulation completed in {end_time - start_time:.2f} seconds')

        error = self.x_history[0, :] - self.reference
        mae = np.mean(np.abs(error))
        print(f'Mean Absolute Error (MAE): {mae:.8f}')

        return {
            'time': self.sim_time,
            'states': self.x_history,
            'estimated_states': self.xk_history,
            'control_inputs': self.mv_history,
            'reference': self.reference,
            'mae': mae,
        }

    def plot_results(self, results=None):
        if results is None:
            results = {
                'time': self.sim_time,
                'states': self.x_history,
                'control_inputs': self.mv_history,
                'reference': self.reference,
            }

        time_vec = results['time']
        states = results['states']
        mv = np.asarray(results['control_inputs'])
        ref = results['reference']

        fig, ax = plt.subplots(6, 1, figsize=(10, 14))

        ax[0].plot(time_vec, states[0, :] * 100, label='Actual Power')
        ax[0].plot(time_vec, ref * 100, '--', label='Desired Power', alpha=0.6)
        ax[0].set_ylabel('Power (SPU)')
        ax[0].set_xlim(0, time_vec[-1])
        ax[0].set_ylim(0, 120)
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(time_vec, states[9, :], label='Fuel')
        ax[1].plot(time_vec, states[10, :], label='Moderator')
        ax[1].plot(time_vec, states[11, :], label='Coolant')
        ax[1].set_ylabel('Temperature (K)')
        ax[1].set_xlim(0, time_vec[-1])
        ax[1].legend()
        ax[1].grid(True)

        if mv.ndim == 1:
            mv = mv.reshape(1, -1)
        for i in range(mv.shape[0]):
            ax[2].plot(time_vec, mv[i], label=f'Drum {i+1}' if mv.shape[0] > 1 else 'Drum')
        ax[2].hlines([0, 180], 0, time_vec[-1], colors='r', linestyles='--', alpha=0.3)
        ax[2].set_ylabel('Drum Position (°)')
        ax[2].set_xlim(0, time_vec[-1])
        if mv.shape[0] > 1:
            ax[2].legend(ncol=4, fontsize=8)
        ax[2].grid(True)

        ax[3].plot(time_vec, (ref * 100) - (states[0, :] * 100))
        ax[3].hlines(0, 0, time_vec[-1], linestyles='--', alpha=0.4)
        ax[3].set_ylabel('Desired - Actual Power (SPU)')
        ax[3].set_xlim(0, time_vec[-1])
        ax[3].grid(True)

        if mv.shape[0] == 1:
            ax[4].plot(time_vec, mv[0], label='Shared Drum')
        else:
            ax[4].plot(time_vec, np.mean(mv, axis=0), label='Mean Drum')
            ax[4].plot(time_vec, np.std(mv, axis=0), '--', label='Std Drum')
        ax[4].set_ylabel('Drum Summary (°)')
        ax[4].set_xlim(0, time_vec[-1])
        ax[4].legend()
        ax[4].grid(True)

        ax[5].plot(time_vec, states[7, :], label='Xe')
        ax[5].plot(time_vec, states[8, :], label='I')
        ax[5].set_ylabel('Number Density')
        ax[5].set_xlim(0, time_vec[-1])
        ax[5].legend()
        ax[5].grid(True)

        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig('./nmpc_oop.png')


def main():
    num_drums = 8
    reactor = ReactorModel(dt=1, num_drums=num_drums)
    ekf = ExtendedKalmanFilter(reactor)
    controller = NonlinearMPC(reactor, prediction_horizon=15, control_horizon=8)
    simulator = Simulator(reactor, ekf, controller, duration=200)
    results = simulator.run_simulation()
    simulator.plot_results(results)


if __name__ == '__main__':
    main()
