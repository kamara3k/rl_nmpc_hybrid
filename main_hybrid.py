#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import gymnasium as gym

# Use local modified Stable-Baselines3 (required for hybrid PPO–NMPC changes) 
# This ensures Python imports the edited SB3 instead of the installed version.
THIS_DIR = Path(__file__).resolve().parent

SB3_REPO = THIS_DIR / "stable-baselines3"

if not (SB3_REPO / "stable_baselines3").exists():
    raise FileNotFoundError(
        f"Editable SB3 not found at {SB3_REPO}. "
        f"Expected stable-baselines3/stable_baselines3/."
    )
sys.path.insert(0, str(SB3_REPO))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecCheckNan
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

import envs
import nmpc


"""
Main training and evaluation script for RL-NMPC control on HOLOS Micro reactor environment.

This script supports:
  - PPO training (pure RL)
  - PPO evaluation
  - Hybrid PPO-NMPC control, where NMPC can override PPO actions. It includes three modes 
  - Optional NMPC-only simulations for running nmpc control system through the gym env

Key features:
  - Supports both HolosMulti (8 drums) and HolosSingle (1 drum) environments
  - Hybrid override mechanism replaces PPO actions with NMPC actions when needed
  - Logs intervention statistics (how often NMPC overrides PPO)
  - Saves results and plots for performance comparison

Usage examples:

  # Train PPO (can include hybrid override if enabled in code)
  python main_hybrid.py --nmpc_mode warmstart --env_kind single --timesteps 200_000 --seed 0 (note that seed must be set for comparison benchmark)
  Also, note that --seed is None in the CLI. Therefore, it must be set 

  # Run NMPC-only simulation (baseline)
  python main_hybrid.py --mode nmpc --episode_length 200

  # Evaluate trained PPO model
  python main_rl_nmpc.py --mode ppo_eval --env_kind single --model_path runs/.../model/ppo_model.zip
"""
class NMPCActionAdapter:

    """
    Adapter for handling action transformations between PPO policy output,
    environment action space, and NMPC controller actions.

    This class ensures consistency when combining reinforcement learning (PPO)
    with model predictive control (NMPC) by:

      - Converting PPO actions (policy space) to environment-compatible actions
      - Converting NMPC actions into the same space expected by the policy
      - Handling scaling, clipping, and bounds enforcement
      - Supporting hybrid control where NMPC can override PPO actions

    This is important for hybrid RL-NMPC control, where actions from different
    sources (learned policy vs. optimization-based controller) must remain
    compatible and physically meaningful.

    Typical usage:
      - PPO proposes an action in policy space
      - NMPC computes an action in control space
      - This adapter maps both into a consistent format before execution
    """
    def __init__(self, num_drums: int, ref_fun_norm, action_space):

        # Initialize adapter with number of drums, reference trajectory, and env action bounds.

        self.num_drums = num_drums
        
        # Store env action bounds
        self.action_space = action_space
        self.act_low = None
        self.act_high = None
        self.act_dim = 1

        # Read env action bounds and convert them to a consistent vector form.
        # lets the adapter handle both scalar and multi-dimensional Box actions uniformly.
        try:
            if hasattr(action_space, "low") and hasattr(action_space, "high"):
                self.act_low = np.asarray(action_space.low, dtype=np.float32).copy()
                self.act_high = np.asarray(action_space.high, dtype=np.float32).copy()
                if self.act_low.ndim == 0:
                    self.act_low = self.act_low.reshape(1)
                    self.act_high = self.act_high.reshape(1)
                self.act_dim = int(self.act_low.shape[0])
        except Exception:
            # fallback: assume scalar [-1, 1]
            self.act_low = np.array([-1.0], dtype=np.float32)
            self.act_high = np.array([1.0], dtype=np.float32)
            self.act_dim = 1

        # Internal reactor model used by EKF and NMPC for state estimation and control.
        self.reactor = nmpc.ReactorModel(dt=1, num_drums=num_drums)
        # State estimator: reconstructs the internal reactor state needed by NMPC from measured outputs.
        self.ekf = nmpc.ExtendedKalmanFilter(self.reactor)
        # NMPC controller used for expert/safety actions.
        # Horizons are fixed here to match the reactor control design used in this script.
        self.controller = nmpc.NonlinearMPC(self.reactor, prediction_horizon=15, control_horizon=8)

        self.ref_fun_norm = ref_fun_norm

    
        # Track previous manipulated variable (degrees) for NMPC warm-start / rate constraints
        self.prev_mv = float(self.reactor.u0)
        # EKF state estimate (initialize near steady state at reset)
        self.x_hat = None  # will be 12-dim vector

        # Store previous measurements to estimate power sensitivity to drum degree dP/d(deg).
        # it is used for one-step safety scaling: before applying a new control action,
        # it approximates how much reactor power will change and reduce the action if it risks violating safety limits.
        self._last_meas_power = None
        self._last_meas_deg = None
        self._last_dP_ddeg = None




    def _delta_deg_to_env_action(self, delta_deg):
        """
        Map a change in drum angle (delta degrees) to the environment action space.
        Env from Leo's configuration uses: real_action = gym_action / 2  (so gym_action = 2*real_action).
        real_action is interpreted as delta-deg/step and is typically clipped to [-0.5, 0.5].
        The HOLOS environment expects actions as per-step increments (delta angles),
        not absolute positions. This function converts the NMPC-computed delta in
        degrees into a normalized action compatible with the Gym action space based on Leo's environment's configuration
        """
        delta_deg = float(np.asarray(delta_deg).reshape(-1)[0])

        # Clip to env's real actuator limit: +/- 0.5 deg/step
        delta_deg = float(np.clip(delta_deg, -0.5, 0.5))

        # Convert to gym action space [-1, 1]
        a0 = 2.0 * delta_deg

        # Clip to bounds (supports custom bounds if not [-1,1])
        lo0 = float(self.act_low[0])
        hi0 = float(self.act_high[0])

        # If bounds are [-1,1], this is direct
        if abs(lo0 + 1.0) < 1e-6 and abs(hi0 - 1.0) < 1e-6:
            a0 = float(np.clip(a0, -1.0, 1.0))
        else:
            # Generic: map [-1,1] -> [lo,hi]
            a0 = lo0 + ((a0 + 1.0) * 0.5) * (hi0 - lo0)
            a0 = float(np.clip(a0, lo0, hi0))

        if self.act_dim > 1:
            a = np.full((self.act_dim,), a0, dtype=np.float32)
            a = np.clip(a, self.act_low, self.act_high)
            return a

        return np.asarray([a0], dtype=np.float32)

    
    # estimate the system state (EKF), based on the current observation
    # compute the optimal control action (NMPC), and convert it into
    # an environment-compatible action (delta drum movement).
    def compute_action(self, obs, t=0.0, info=None):
        
        """
        Compute the control action using NMPC based on the current observation.

        This method implements the NMPC side of the hybrid RL–NMPC framework.
        It performs the following steps:

           1. Extract measured variables from the environment observation
              (e.g., power, drum angles, reference trajectory).
           2. Update the EKF to estimate the full reactor state.
           3. Solve the NMPC optimization problem to obtain the optimal control
              input (absolute drum angle in degrees).
           4. Convert the absolute NMPC action into a delta (increment), since
              the environment expects per-step changes in drum angle.
           5. Optionally apply safety scaling using empirical sensitivity
               (dP/ddeg) to prevent unsafe power excursions.
           6. Map the delta control to the environment action space.

        Returns:
          action (np.ndarray): normalized action compatible with the Gym environment
        """
        info = info or {}

        # Extract current drum angle in degrees 
        cur_deg = 77.8  # fallback nominal drum angle if observation parsing fails
        try:
            if isinstance(obs, dict):
                if "drum_angles" in obs:
                    # stored as fraction [0,1], convert to degrees
                    cur_deg = float(np.asarray(obs["drum_angles"]).reshape(-1)[0]) * 180.0
                elif "drum_angle" in obs:
                    cur_deg = float(np.asarray(obs["drum_angle"]).reshape(-1)[0]) * 180.0
        except Exception:
            pass

        # Build/update a consistent 12-state estimate for the NMPC model.
        # env observation only provides power and drum angle; NMPC plant needs full state.
        # maintain an EKF estimate using measured power.
        y_meas = None
        try:
            if isinstance(obs, dict) and "power" in obs:
                y_meas = float(np.asarray(obs["power"]).reshape(-1)[0])
        except Exception:
            y_meas = None

        # Initialize x_hat if needed (steady-state-ish)
        if self.x_hat is None:
            try:
                n0 = 1.0 if y_meas is None else float(y_meas)
                # steady precursor concentrations: c_i = beta_i/lambda_i * n
                c0 = (self.reactor.betas / self.reactor.lambdas) * n0
                self.x_hat = np.zeros(12, dtype=float)
                self.x_hat[0] = n0
                self.x_hat[1:7] = c0
                self.x_hat[7] = float(getattr(self.reactor, "Xe0", 0.0))
                self.x_hat[8] = float(getattr(self.reactor, "I0", 0.0))
                self.x_hat[9] = float(getattr(self.reactor, "Tf0", 0.0))
                self.x_hat[10] = float(getattr(self.reactor, "Tm0", 0.0))
                self.x_hat[11] = float(getattr(self.reactor, "Tc0", 0.0))
            except Exception:
                self.x_hat = np.zeros(12, dtype=float)

        # If caller did not supply y_meas_local, try to infer measured power from observation dict
        y_meas_local = y_meas
        
        # Update the EKF using measured power and current drum angle to obtain
        # the latest full-state estimate required by NMPC.
        if y_meas_local is not None:
            try:
                Ts = float(getattr(self.reactor, "Ts", 1.0))
                self.x_hat = self.ekf.update(self.x_hat, y_meas_local, float(cur_deg), Ts)
            except Exception:
                pass

        x = self.x_hat

        # Reference for NMPC:

        # HOLOS env (Leo) observation includes next_desired_power (normalized 0..1) and reward is computed
        # against desired power after the env increments its internal time. So next_desired_power should
        # align NMPC with the reward shaping. Fall back to ref_fun_norm(t) if not present.
        r = None
        try:
            if isinstance(obs, dict) and "next_desired_power" in obs:
                r = float(np.asarray(obs["next_desired_power"]).reshape(-1)[0])
        except Exception:
            r = None
        if r is None:
            r = float(self.ref_fun_norm(t))

        # Deadband: if power is already close to the next target, hold the current angle
        # to avoid unnecessary moves and overshoot.
        if y_meas_local is not None and isinstance(obs, dict) and "next_desired_power" in obs:
            r_safe = float(np.asarray(obs["next_desired_power"]).reshape(-1)[0])
            if abs(float(y_meas_local) - r_safe) < 0.005:
                self.prev_mv = float(cur_deg)
                self._last_meas_power = float(y_meas_local)
                self._last_meas_deg = float(cur_deg)
                return self._delta_deg_to_env_action(0.0)
    

        # Solve NMPC for the next absolute drum-angle target in degrees.
        u_target_deg = self.controller.calculate_control(x, self.prev_mv, r)

        # Store the NMPC output as the previous manipulated variable for the next control step.
        try:
            self.prev_mv = float(np.asarray(u_target_deg).reshape(-1)[0])
        except Exception:
            self.prev_mv = float(u_target_deg)

        # If NMPC returns an invalid value, fall back to holding the current drum angle.
        try:
            if not np.isfinite(self.prev_mv):
                self.prev_mv = float(cur_deg)
        except Exception:
            self.prev_mv = float(cur_deg)

        # HOLOS envs (Leo) expects per-step angle increments, so convert absolute angle
        # into a delta relative to the current drum position.
        delta_deg = self.prev_mv - float(cur_deg)

        # If NMPC returns deos not give action but tracking error is noticeable,
        # then apply a small one-step action in the correct direction.
        try:
            if y_meas_local is not None:
                _err = float(y_meas_local) - float(r)
                if (abs(_err) > 0.01) and (abs(float(delta_deg)) < 1e-9):
                    delta_deg = -1.0 if _err > 0 else 1.0
                    self.prev_mv = float(cur_deg) + float(delta_deg)
        except Exception:
            pass


        # --- One-step safety scaling ---
        # Env has a hard violation when |power - desired| > 0.05. use a predictive method to reduce the proposed delta (change)
        # if it will cause power to violate tracking band on the next step
        
        try:
            if y_meas_local is not None:
                # current target used for reward is next desired power
                r_safe = r
                if isinstance(obs, dict) and "next_desired_power" in obs:
                    r_safe = float(np.asarray(obs["next_desired_power"]).reshape(-1)[0])

                # estimate local dP/ddeg from last step (power change vs applied delta)
                dP_ddeg = None
                if (self._last_meas_power is not None) and (self._last_meas_deg is not None):
                    dp = float(y_meas_local) - float(self._last_meas_power)
                    ddeg = float(cur_deg) - float(self._last_meas_deg)
                    if abs(ddeg) > 1e-6:
                        dP_ddeg = dp / ddeg
                    # persist last valid sensitivity
                    #self._last_dP_ddeg = float(dP_ddeg)

                #if dP_ddeg is None and self._last_dP_ddeg is not None:
                    #dP_ddeg = float(self._last_dP_ddeg)

                if dP_ddeg is not None:
                    # Predict next power if we apply this delta (first-order)
                    # NOTE: env clips to +/-0.5 deg/step anyway; we apply scaling before that.
                    # baseline drift from last step (inertia / delayed response)
                    dp_last = 0.0
                    if self._last_meas_power is not None:
                        dp_last = float(y_meas_local) - float(self._last_meas_power)

                                        # Baseline-only next-step prediction (no additional control)
                    p_pred0 = float(y_meas_local) + dp_last

                    # Predict next power with baseline drift + control effect (first-order)
                    p_pred = float(y_meas_local) + dp_last + float(dP_ddeg) * float(delta_deg)

                    # Allowed band around desired (keep some slack)
                    band = 0.05
                    slack = 0.010  # keep 1.0% margin
                    lo = float(r_safe) - (band - slack)
                    hi = float(r_safe) + (band - slack)

                    # If baseline drift alone is about to violate, do NOT push further in the same direction.
                    if p_pred0 < lo and delta_deg < 0.0:
                        delta_deg = 0.0
                    elif p_pred0 > hi and delta_deg > 0.0:
                        delta_deg = 0.0


                    if p_pred < lo or p_pred > hi:
                        # compute max safe delta under linear model
                        # clamp predicted within [lo, hi]
                        p_clamp = min(max(p_pred, lo), hi)
                        if abs(dP_ddeg) > 1e-9:
                            delta_safe = (p_clamp - float(y_meas_local)) / float(dP_ddeg)
                            # only scale down magnitude (keep direction)
                            if abs(delta_safe) < abs(delta_deg):
                                delta_deg = float(delta_safe)
        except Exception:
            pass

        # Update memory for next control stepto estimate sensitivity 
        try:
            if y_meas_local is not None:
                self._last_meas_power = float(y_meas_local)
                self._last_meas_deg = float(cur_deg)
        except Exception:
            pass

        # Map delta_deg -> env action
        return self._delta_deg_to_env_action(delta_deg)



class InterventionLogger(BaseCallback):
    """log NMPC override rate vs timesteps.
    This callback records how often the NMPC controller overrides
    the PPO policy, as a function of training timesteps.
    Note:
    Requires patched SB3 OnPolicyAlgorithm to increment:
      - self._nmpc_override_env_steps
      - self._nmpc_total_env_steps
    """
    def __init__(self, log_path: Path, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.log_path = Path(log_path)
        self.log_freq = int(log_freq)
        self.ts = []
        self.override_rates = []
        self.overrides = []
        self.totals = []

    def _on_step(self) -> bool:
        if self.log_freq <= 0:
            return True
        if (self.num_timesteps % self.log_freq) != 0:
            return True
        overrides = int(getattr(self.model, "_nmpc_override_env_steps", 0) or 0)
        totals = int(getattr(self.model, "_nmpc_total_env_steps", 0) or 0)
        rate = (100.0 * overrides / totals) if totals > 0 else 0.0
        self.ts.append(int(self.num_timesteps))
        self.override_rates.append(float(rate))
        self.overrides.append(overrides)
        self.totals.append(totals)
        return True

    def _on_training_end(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(self.log_path),
            timesteps=np.asarray(self.ts, dtype=np.int64),
            override_rate=np.asarray(self.override_rates, dtype=np.float32),
            overrides=np.asarray(self.overrides, dtype=np.int64),
            totals=np.asarray(self.totals, dtype=np.int64),
        )


# -----------------------------
# Reference profile (SPU 0-100)
# -----------------------------
_REF_TIMES = [0, 25, 35, 50, 60, 80, 125, 135, 150, 175, 200]
_REF_VALUES_SPU = [100, 100, 80, 70, 70, 50, 50, 60, 60, 90, 90]

def make_reference_profile():
    # Returns SPU in [0, 100]; envs.py divides by 100 internally.
    return interp1d(
        _REF_TIMES,
        np.asarray(_REF_VALUES_SPU, dtype=float),
        bounds_error=False,
        fill_value=(float(_REF_VALUES_SPU[0]), float(_REF_VALUES_SPU[-1])),
    )


# -----------------------------
# Run directories
# -----------------------------
def create_run_dirs(run_name: str):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / f"{timestamp}_{run_name}"
    model_dir = run_dir / "model"
    graph_dir = run_dir / "graphs"
    log_dir = run_dir / "logs"

    model_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, model_dir, graph_dir, log_dir


# -----------------------------
# Environment factories
# -----------------------------
def make_env_factory(env_kind: str, profile, episode_length: int, run_path: Path | None, train_mode: bool, noise: float):
    """
    Returns a callable that creates a new env instance.
    """
    def _init():
        if env_kind == "multi":
            env = envs.HolosMulti(
                profile=profile,
                episode_length=episode_length,
                run_path=run_path,
                train_mode=train_mode,
                noise=noise,
                debug=False,
                valid_maskings=(0,),
                symmetry_reward=False,
            )
            #env = ActionClipWrapper(env)
            #env = SanitizeObsRewardWrapper(env)
            return env
        elif env_kind == "single":
            env = envs.HolosSingle(
                profile=profile,
                episode_length=episode_length,
                run_path=run_path,
                train_mode=train_mode,
                noise=noise,
                debug=False,
                valid_maskings=(0,),
            )
            #env = ActionClipWrapper(env)
            #env = SanitizeObsRewardWrapper(env)
            return env
        else:
            raise ValueError(f"Unknown env_kind: {env_kind}")
    return _init


# -----------------------------
# Plotting: NMPC-style (one figure)
# -----------------------------
def plot_nmpc_style(df, save_path: Path):
    """
    for plotting the power and states values:
      1) Power (Actual vs Desired)
      2) Temperatures (Fuel/Moderator/Coolant)
      3) Drum position with 0/180 bounds
      4) Desired - Actual power
      5) Drum position (mean) (clean)
      6) Xe and I number density
    """
    t = df["time"].to_numpy()

    desired_power = df["desired_power"].to_numpy() * 100.0
    actual_power = df["actual_power"].to_numpy() * 100.0


    measured_power = df["measured_power"].to_numpy() * 100.0 if "measured_power" in df.columns else None

    Tf = df["Tf"].to_numpy()
    Tm = df["Tm"].to_numpy()
    Tc = df["Tc"].to_numpy()

    drum_cols = [c for c in df.columns if c.startswith("drum_")]
    drums = df[drum_cols].to_numpy()
    drum_1 = df["drum_1"].to_numpy() if "drum_1" in df.columns else drums[:, 0]
    drum_mean = drums.mean(axis=1)

    power_err = desired_power - actual_power

    Xe = df["Xe"].to_numpy()
    I = df["I"].to_numpy()

    fig, ax = plt.subplots(6, 1, figsize=(10, 14), sharex=True)

    # 1) Power
    ax[0].plot(t, actual_power, label="Actual Power")
    ax[0].plot(t, desired_power, "--", label="Desired Power")
    if measured_power is not None:
    
        pass
    ax[0].set_ylabel("Power (SPU)")
    ax[0].grid(True)
    ax[0].legend(loc="lower left")

    # 2) Temperatures
    ax[1].plot(t, Tf, label="Fuel")
    ax[1].plot(t, Tm, label="Moderator")
    ax[1].plot(t, Tc, label="Coolant")
    ax[1].set_ylabel("Temperature (°C)")
    ax[1].grid(True)
    ax[1].legend(loc="upper center", ncol=3)

    # 3) Drum position with bounds
    ax[2].plot(t, drum_1, label="Drum Position")
    ax[2].axhline(0.0, linestyle="--", color="red", alpha=0.5)
    ax[2].axhline(180.0, linestyle="--", color="red", alpha=0.5)
    ax[2].set_ylabel("Drum Position (°)")
    ax[2].grid(True)

    # 4) Desired - Actual
    ax[3].plot(t, power_err)
    ax[3].axhline(0.0, linestyle="--", alpha=0.5)
    ax[3].set_ylabel("Desired - Actual Power (SPU)")
    ax[3].grid(True)

    # 5) Drum mean (clean)
    ax[4].plot(t, drum_mean)
    ax[4].set_ylabel("Drum Position (°)")
    ax[4].grid(True)

    # 6) Xe and I
    ax[5].plot(t, Xe, label="Xe")
    ax[5].plot(t, I, label="I")
    ax[5].set_ylabel("Number Density")
    ax[5].set_xlabel("Time (s)")
    ax[5].grid(True)
    ax[5].legend(loc="center right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)

def plot_learning_curves(log_dir: Path, graph_dir: Path, title: str = "Learning Curves"):
    """Plots:
      1) Training episode return vs (approx) timesteps from VecMonitor CSV
      2) Evaluation return (PPO-only) vs timesteps from EvalCallback evaluations.npz
      3) Override rate (%) vs timesteps from interventions.npz
    """
    graph_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    # 1) Training returns
    train_csv = log_dir / "train_monitor.csv"
    if train_csv.exists():
        try:
            df = pd.read_csv(train_csv, comment="#")
            if "r" in df.columns:
                ts = df["l"].cumsum() if "l" in df.columns else np.arange(len(df))
                ax[0].plot(np.asarray(ts), df["r"].to_numpy())
                ax[0].set_ylabel("Train episode return")
                ax[0].grid(True)
        except Exception as e:
            ax[0].text(0.01, 0.5, f"Failed to read {train_csv}: {e}", transform=ax[0].transAxes)
    else:
        ax[0].text(0.01, 0.5, "No train_monitor.csv found", transform=ax[0].transAxes)

    # 2) Eval returns
    eval_npz = log_dir / "eval" / "evaluations.npz"
    if eval_npz.exists():
        data = np.load(eval_npz, allow_pickle=True)
        timesteps = data.get("timesteps")
        results = data.get("results")
        if timesteps is not None and results is not None:
            mean_r = results.mean(axis=1)
            std_r = results.std(axis=1)
            ax[1].plot(timesteps, mean_r)
            ax[1].fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.2)
            ax[1].set_ylabel("Eval return (PPO-only)")
            ax[1].grid(True)
    else:
        ax[1].text(0.01, 0.5, "No eval/evaluations.npz found", transform=ax[1].transAxes)

    # 3) Override rate
    int_npz = log_dir / "interventions.npz"
    if int_npz.exists():
        data = np.load(int_npz, allow_pickle=True)
        ts = data.get("timesteps")
        rate = data.get("override_rate")
        if ts is not None and rate is not None:
            ax[2].plot(ts, rate)
            ax[2].set_ylabel("Override rate (%)")
            ax[2].set_xlabel("Timesteps")
            ax[2].grid(True)
    else:
        ax[2].text(0.01, 0.5, "No interventions.npz found", transform=ax[2].transAxes)

    fig.suptitle(title)
    fig.tight_layout()
    out_path = graph_dir / "learning_curves.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved learning curves to {out_path}")


# -----------------------------
# PPO train / eval
# -----------------------------
def train_ppo(env_kind: str, run_dir: Path, model_dir: Path, graph_dir: Path, log_dir: Path,
              timesteps: int, episode_length: int, n_envs: int, noise: float,
              seed: int | None,
              num_drums: int,
              nmpc_mode: str, nmpc_warmup_steps: int, nmpc_mix_alpha: float,
              nmpc_mix_alpha_final: float, nmpc_mix_anneal_steps: int,
              nmpc_shield_kind: str, nmpc_shield_power_err: float, nmpc_shield_drum_margin: float):

    profile = make_reference_profile()

    # pass run_dir to the env so envs.py can save CSVs into the run folder.
    vec_env = DummyVecEnv([
        make_env_factory(env_kind, profile, episode_length, run_dir, True, noise)
        for _ in range(n_envs)
    ])
    vec_env = VecMonitor(vec_env, filename=str(Path(log_dir) / "train_monitor.csv"))
    vec_env = VecCheckNan(vec_env, raise_exception=True)


    model = PPO(
        policy="MultiInputPolicy",  
        env=vec_env,
        verbose=1,
        seed=seed,
        
    )

    
    profile_spu = make_reference_profile()
    def ref_fun_norm(t):
        return np.asarray(profile_spu(t), dtype=float) / 100.0
    
    
    # Attach NMPC adapter (converts degrees -> env action space automatically)
    model.nmpc_controller = NMPCActionAdapter(
        num_drums=num_drums,
        ref_fun_norm=ref_fun_norm,
        action_space=vec_env.action_space,
    )
    # attach nmpc_mode to the ppo model, which can be warmstart, mix or shield
    model.nmpc_mode = nmpc_mode

    # Warmstart / mix/shield parameters
    model.nmpc_warmup_steps = nmpc_warmup_steps
    model.nmpc_mix_alpha = nmpc_mix_alpha
    model.nmpc_mix_alpha_final = nmpc_mix_alpha_final
    model.nmpc_mix_anneal_steps = nmpc_mix_anneal_steps

    # BC pretrain settings (used only when nmpc_mode == "warmstart")
    #model.nmpc_bc_pretrain = (nmpc_mode == "warmstart" and nmpc_warmup_steps > 0)
    #model.nmpc_bc_epochs = 10
    #model.nmpc_bc_batch = 256
    #model.nmpc_bc_lr = 3e-4
    model.nmpc_shield_kind = nmpc_shield_kind
    model.nmpc_shield_power_err = nmpc_shield_power_err
    model.nmpc_shield_drum_margin = nmpc_shield_drum_margin

    print("[INFO] Hybrid settings:",
          "mode=", model.nmpc_mode,
          "warmup_steps=", model.nmpc_warmup_steps,
          "mix_alpha0=", model.nmpc_mix_alpha,
          "mix_alphaF=", getattr(model, "nmpc_mix_alpha_final", model.nmpc_mix_alpha),
          "mix_anneal_steps=", getattr(model, "nmpc_mix_anneal_steps", 0),
          "shield_kind=", getattr(model, "nmpc_shield_kind", "hybrid"))
    print("[INFO] Env action_space:", vec_env.action_space)

    # ---- Callbacks: PPO-only evaluation + intervention logging ----
    eval_env = make_env_factory(env_kind, profile, episode_length, None, False, 0.0)()
    (Path(log_dir) / "eval").mkdir(parents=True, exist_ok=True)
    eval_env = Monitor(eval_env, filename=str(Path(log_dir) / "eval" / "eval_monitor.csv"))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(Path(log_dir) / "eval"),
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )
    int_cb = InterventionLogger(log_path=Path(log_dir) / "interventions.npz", log_freq=5000)
    model.learn(total_timesteps=timesteps, callback=[eval_cb, int_cb])


    model_path = model_dir / "ppo_model.zip"
    model.save(model_path)
    print(f"[INFO] PPO model saved to {model_path}")

    # Plot learning curves (training return, eval return, override rate)
    try:
        plot_learning_curves(log_dir=Path(log_dir), graph_dir=graph_dir, title=f"{env_kind} | {nmpc_mode}")
    except Exception as e:
        print("[WARN] Failed to plot learning curves:", e)

    return model, model_path



def evaluate_ppo(
    env_kind: str,
    model: PPO,
    graph_dir: Path,
    episode_length: int,
    noise: float = 0.0,
    deterministic: bool = True,
    eval_hybrid: bool = False,
    log_dir: Path | None = None,
):
    """
    Run one evaluation episode and save the trajectory summary plot.

    Evaluation can be done in either:
     - PPO-only mode, or
     - hybrid mode, where NMPC override/mixing is applied using the model's
        stored hybrid settings.

    This function does not train the model. It only rolls out one episode,
    retrieves the recorded trajectory history from the underlying HOLOS env,
    and saves the summary figure.

    """
    profile = make_reference_profile()
    env = make_env_factory(env_kind, profile, episode_length, None, False, noise)()

    obs, info = env.reset()
    terminated = truncated = False

    # Optional: apply NMPC hybrid override during evaluation so plots reflect hybrid behavior
    hybrid_enabled = bool(eval_hybrid) and (
        hasattr(model, "nmpc_controller") and model.nmpc_controller is not None and
        getattr(model, "nmpc_mode", "off") != "off"
    )
    nmpc_mode = getattr(model, "nmpc_mode", "off")
    warmup_steps = int(getattr(model, "nmpc_warmup_steps", 0) or 0)
    alpha0 = float(getattr(model, "nmpc_mix_alpha", 1.0))
    alphaF = float(getattr(model, "nmpc_mix_alpha_final", alpha0))
    anneal_steps = int(getattr(model, "nmpc_mix_anneal_steps", 0) or 0)
    # mix_alpha will be computed per-step if annealing is enabled
    mix_alpha = max(0.0, min(1.0, alpha0))

    # Reset adapter internal prev_mv at start of episode (if available)
    if hybrid_enabled and hasattr(model.nmpc_controller, "prev_mv") and hasattr(model.nmpc_controller, "reactor"):
        try:
            model.nmpc_controller.prev_mv = float(model.nmpc_controller.reactor.u0)
        except Exception:
            pass

    step_idx = 0
    while not (terminated or truncated):
        # PPO proposal
        action_ppo, _ = model.predict(obs, deterministic=deterministic)

        if not hybrid_enabled:
            action = action_ppo
        else:
            
            t_now = 0.0
            if isinstance(info, dict) and ("time" in info):
                try:
                    t_now = float(info.get("time", 0.0))
                except Exception:
                    t_now = float(step_idx)
            else:
                t_now = float(step_idx)

            # decide whether to override
            use_nmpc = (nmpc_mode == "always")
            if nmpc_mode == "warmstart":
                use_nmpc = (step_idx < warmup_steps)
            elif nmpc_mode == "shield":
                
                use_nmpc = bool(info.get("constraint_violation", False) or info.get("unsafe", False) or info.get("nmpc_override", False))
            elif nmpc_mode == "mix":
                use_nmpc = True

            if use_nmpc:
                action_nmpc = model.nmpc_controller.compute_action(obs, t=t_now, info=info)
                if nmpc_mode == "mix":
                    # Mix in env action space (both already scaled)
                    if anneal_steps > 0:
                        frac = float(step_idx) / float(anneal_steps)
                        frac = max(0.0, min(1.0, frac))
                        mix_alpha = alpha0 + frac * (alphaF - alpha0)
                        mix_alpha = max(0.0, min(1.0, float(mix_alpha)))
                    action = mix_alpha * np.asarray(action_nmpc, dtype=np.float32) + (1.0 - mix_alpha) * np.asarray(action_ppo, dtype=np.float32)
                else:
                    action = action_nmpc
            else:
                action = action_ppo

        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1

    #df = get_history_df(env)
    base_env = env.multi_env if hasattr(env, "multi_env") else env
    base_env.render()
    df = base_env.history

    fig_path = graph_dir / "episode_summary.png"
    plot_nmpc_style(df, fig_path)
    print(f"[INFO] Saved combined plot to {fig_path}")


# -----------------------------
# NMPC-only run (nmpc.py)

# -----------------------------
def run_nmpc_only(graph_dir: Path, duration: int, num_drums: int):
    """
    Run a standalone NMPC-only simulation using nmpc.py.

    This function builds the reactor model, EKF, and NMPC controller from nmpc.py,
    then runs a closed-loop NMPC simulation using the same reference profile as the
    PPO/HOLOS experiments. The reference from this main script is converted from
    SPU scale (0-100) to normalized scale (0-1) before being passed to the simulator.
    """
    # Reference profile in this main script is SPU (0–100).
    # nmpc.py expects a normalized reference (0–1), so we scale here.
    profile_spu = make_reference_profile()

    def ref_fun_norm(t):
        return np.asarray(profile_spu(t), dtype=float) / 100.0

    reactor = nmpc.ReactorModel(dt=1, num_drums=num_drums)
    ekf = nmpc.ExtendedKalmanFilter(reactor)
    controller = nmpc.NonlinearMPC(reactor, prediction_horizon=15, control_horizon=8)

    # Pass ref override so NMPC follows the same reference used in PPO runs
    sim = nmpc.Simulator(
        reactor, ekf, controller,
        duration=duration,
        ref_fun_override=ref_fun_norm,
    )

    results = sim.run_simulation()
    sim.plot_results(results)

    # nmpc.py saves its own default file; also save into this run folder
    out_path = graph_dir / "nmpc_oop.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved NMPC plot to {out_path}")



# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="ppo", choices=["ppo", "ppo_eval", "nmpc"])
    p.add_argument("--env_kind", type=str, default="single", choices=["single", "multi"])
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--episode_length", type=int, default=200)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_path", type=str, default=None, help="For --mode ppo_eval")
    p.add_argument("--num_drums", type=int, default=None, help="enter numb_drums 1 or 8")
    p.add_argument("--nmpc_mode", type=str, default="off",
              choices=["off", "always", "warmstart", "mix", "shield"])
    p.add_argument("--nmpc_warmup_steps", type=int, default=1000)
    p.add_argument("--nmpc_mix_alpha", type=float, default=0.6)
    p.add_argument("--nmpc_mix_alpha_final", type=float, default=0.0,
                  help="For mix mode with annealing: final alpha at end of anneal window (default 0.0)")
    p.add_argument("--nmpc_mix_anneal_steps", type=int, default=1_000,
                  help="If >0 and mode=mix: linearly anneal alpha from nmpc_mix_alpha to nmpc_mix_alpha_final over these timesteps")
    p.add_argument("--nmpc_shield_kind", type=str, default="hybrid", choices=["info", "state", "hybrid"],
                  help="Shield trigger: env info flags, state-based checks, or both")
    p.add_argument("--nmpc_shield_power_err", type=float, default=0.05,
                  help="State-based shield: |power-desired| threshold (normalized 0..1)")
    p.add_argument("--nmpc_shield_drum_margin", type=float, default=0.02,
                  help="State-based shield: drum angle margin near 0/1 bounds in obs space")
    p.add_argument("--eval_hybrid", action="store_true",
                  help="If set, apply NMPC override during evaluation. Default: evaluate PPO-only.")

    return p.parse_args()


def main():
    args = parse_args()

    # If not specified, match env_kind convention: single=1 drum, multi=8 drums
    if args.num_drums is None:
        args.num_drums = 8 if args.env_kind == "multi" else 1

    run_name = f"{args.mode}_{args.env_kind}_drums{args.num_drums}"
    run_dir, model_dir, graph_dir, log_dir = create_run_dirs(run_name)

    # Save config for reproducibility
    (run_dir / "config.txt").write_text(
        f"mode={args.mode}\n"
        f"env_kind={args.env_kind}\n"
        f"timesteps={args.timesteps}\n"
        f"episode_length={args.episode_length}\n"
        f"n_envs={args.n_envs}\n"
        f"noise={args.noise}\n"
        f"seed={args.seed}\n"
        f"model_path={args.model_path}\n"
        f"num_drums={args.num_drums}\n"

    )

    if args.mode == "ppo":
        
        model, model_path = train_ppo(
            env_kind=args.env_kind,
            run_dir=run_dir,
            model_dir=model_dir,
            graph_dir=graph_dir,
            log_dir=log_dir,
            timesteps=args.timesteps,
            episode_length=args.episode_length,
            n_envs=args.n_envs,
            noise=args.noise,
            seed=args.seed,
            num_drums=args.num_drums,
            nmpc_mode=args.nmpc_mode,
            nmpc_warmup_steps=args.nmpc_warmup_steps,
            nmpc_mix_alpha=args.nmpc_mix_alpha,
            nmpc_mix_alpha_final=args.nmpc_mix_alpha_final,
            nmpc_mix_anneal_steps=args.nmpc_mix_anneal_steps,
            nmpc_shield_kind=args.nmpc_shield_kind,
            nmpc_shield_power_err=args.nmpc_shield_power_err,
            nmpc_shield_drum_margin=args.nmpc_shield_drum_margin,
        )


        # Evaluate immediately after training (single combined NMPC-style figure)
        evaluate_ppo(
            env_kind=args.env_kind,
            model=model,
            graph_dir=graph_dir,
            log_dir=log_dir,
            episode_length=args.episode_length,
            noise=args.noise,
            deterministic=True,
            eval_hybrid=args.eval_hybrid,
        )

    elif args.mode == "ppo_eval":
        if args.model_path is None:
            raise ValueError("--model_path is required for --mode ppo_eval")

        model = PPO.load(args.model_path)

        profile_spu = make_reference_profile()

        def ref_fun_norm(t):
            return np.asarray(profile_spu(t), dtype=float) / 100.0

        eval_env_tmp = make_env_factory(
            args.env_kind, profile_spu, args.episode_length, None, False, args.noise
        )()

        model.nmpc_controller = NMPCActionAdapter(
            num_drums=args.num_drums,
            ref_fun_norm=ref_fun_norm,
            action_space=eval_env_tmp.action_space,
        )

        model.nmpc_mode = args.nmpc_mode
        model.nmpc_warmup_steps = args.nmpc_warmup_steps
        model.nmpc_mix_alpha = args.nmpc_mix_alpha
        model.nmpc_mix_alpha_final = args.nmpc_mix_alpha_final
        model.nmpc_mix_anneal_steps = args.nmpc_mix_anneal_steps
        model.nmpc_shield_kind = args.nmpc_shield_kind
        model.nmpc_shield_power_err = args.nmpc_shield_power_err
        model.nmpc_shield_drum_margin = args.nmpc_shield_drum_margin

        evaluate_ppo(
            env_kind=args.env_kind,
            model=model,
            graph_dir=graph_dir,
            log_dir=log_dir,
            episode_length=args.episode_length,
            noise=args.noise,
            deterministic=True,
            eval_hybrid=args.eval_hybrid,
        )

    elif args.mode == "nmpc":
        run_nmpc_only(graph_dir, duration=args.episode_length, num_drums=args.num_drums)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
