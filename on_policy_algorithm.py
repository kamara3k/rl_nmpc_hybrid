import sys
import time
import warnings
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[tuple[type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        #kamal_start

        self.nmpc_controller = None
        self.nmpc_mode = "off"
        self._last_infos = None

        # Hybrid attributes for warmstart and mix modes 
        self.nmpc_warmup_steps = 0          # if > 0 and mode="warmstart", NMPC runs for the first N steps
        self.nmpc_mix_alpha = 1.0           # if mode="mix": final = alpha*NMPC + (1-alpha)*PPO
        self.nmpc_mix_alpha_final = 0.0     # if annealing enabled: alpha -> final over nmpc_mix_anneal_steps
        self.nmpc_mix_anneal_steps = 0      # if > 0 and mode="mix": linearly anneal alpha with num_timesteps 

        # additional hybrid attributes for shield mode
        self.nmpc_shield_kind = "hybrid"   # can be either of : 'info', 'state', 'hybrid'
        self.nmpc_shield_power_err = 0.05   # normalized power error threshold (e.g. 0.05 = 5%)
        self.nmpc_shield_drum_margin = 0.02 # normalized drum margin in obs space (0..1)

        self.nmpc_debug_once = False        # optional: print keys once for debuugging 

        #kaml_ends

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )
    
    #kamal_start 
    

    def _maybe_override_with_nmpc(self, obs, actions, infos_prev):
        """
        Optionally replace PPO actions with NMPC actions before stepping the env using env.step().

        Supports obs being either:
          - np.ndarray with shape [n_envs, ...]
          - dict of np.ndarray (VecEnv Dict obs): {k: array([n_envs, ...]), ...}
        """
        if not hasattr(self, "nmpc_controller") or self.nmpc_controller is None:
            return actions

        mode = getattr(self, "nmpc_mode", "off")
        if mode == "off":
            return actions

        # Warmstart: NMPC for first N global timesteps, then PPO
        warmup_steps = int(getattr(self, "nmpc_warmup_steps", 0) or 0)
        if mode == "warmstart" and warmup_steps > 0 and self.num_timesteps >= warmup_steps:
            return actions

        # Determine n_envs, act_dim  to ensure compliance with the SB3. Also safe-proof in case. 
        if actions.ndim == 1:
            n_envs = actions.shape[0]
            act_dim = 1
            actions_2d = actions.reshape(-1, 1)
        else:
            n_envs = actions.shape[0]
            act_dim = actions.shape[1]
            actions_2d = actions

        # normalization: infos_prev is a list of length n_envs, where each element is a dict—one per parallel environment. To ensure compliance with what is comming from Leo's env design
        if infos_prev is None:
            infos_prev = [{} for _ in range(n_envs)]
        else:
            if len(infos_prev) < n_envs:
                infos_prev = list(infos_prev) + [{} for _ in range(n_envs - len(infos_prev))]
            elif len(infos_prev) > n_envs:
                infos_prev = infos_prev[:n_envs]

        def obs_i_from(obs_all, i):
            # Dict obs: slice each key by env index i; obs-all is the observation from all environments
            if isinstance(obs_all, dict):
                return {k: (v[i] if hasattr(v, "__len__") else v) for k, v in obs_all.items()}
            # Array obs
            return obs_all[i]

        # Optional one-time debug
        if getattr(self, "nmpc_debug_once", False) is False:
            # flip it so it only prints once if enabled later
            pass

        # helper: call NMPC controller  for the override
        def call_nmpc(obs_i, t_i, info_i):
            ctrl = self.nmpc_controller
            if hasattr(ctrl, "compute_action"):
                return ctrl.compute_action(obs=obs_i, t=t_i, info=info_i)
            raise AttributeError("NMPC controller must expose compute_action() or predict() or get_action()")
        
        if mode == "mix":
            alpha0 = float(getattr(self, "nmpc_mix_alpha", 1.0))
            alphaF = float(getattr(self, "nmpc_mix_alpha_final", alpha0))
            anneal_steps = int(getattr(self, "nmpc_mix_anneal_steps", 0) or 0)

            if anneal_steps > 0:
                frac = float(self.num_timesteps) / float(anneal_steps)
                frac = max(0.0, min(1.0, frac))
                alpha = alpha0 + frac * (alphaF - alpha0)
            
            else:
                alpha = alpha0

            alpha = max(0.0, min(1.0, float(alpha)))

            if alpha <= 1e-12: 
                return actions if actions.ndim > 1 else actions.reshape(-1)   
            
        # hybrid modes. always is only for sanity check to confirm that nmopc is called and if used, it calls nmpc throughout the total timesteps
        if mode in ("always", "warmstart", "mix", "shield"):
            new_actions = []

            for i in range(n_envs):
                info_i = infos_prev[i] or {}
                t_i = info_i.get("time", 0.0)

                # Shield mode: only override based on the constraints provided.
                
                # Shield mode: decide whether PPO action is allowed or NMPC must take over.
                # it supports three variants depending on how or what is used to trigger the shield mode:
                #  - 'info': rely only on env-provided flags in info_i
                #  - 'state': compute safety/need from current obs + PPO action
                #  - 'hybrid': (info OR state)
                if mode == "shield":
                    shield_kind = str(getattr(self, "nmpc_shield_kind", "hybrid")).lower()

                    # --- Info-based need ---
                    need_info = (
                        bool(info_i.get("constraint_violation", False)) or
                        bool(info_i.get("unsafe", False)) or
                        bool(info_i.get("nmpc_override", False))
                    )

                    # --- State-based need ---
                    need_state = False
                    if shield_kind in ("state", "hybrid"):
                        try:
                            obs_i = obs_i_from(obs, i)
                            # Extract power and desired power 
                            p = None
                            d = None
                            if isinstance(obs_i, dict):
                                if "power" in obs_i:
                                    p = float(np.asarray(obs_i["power"]).reshape(-1)[0])
                                # prefer env-provided desired_power (current), else fall back to next_desired_power
                                if "desired_power" in info_i:
                                    d = float(info_i.get("desired_power"))
                                elif "next_desired_power" in obs_i:
                                    d = float(np.asarray(obs_i["next_desired_power"]).reshape(-1)[0])
                            # Power-error threshold
                            if p is not None and d is not None:
                                thr = float(getattr(self, "nmpc_shield_power_err", 0.05))
                                if abs(p - d) > thr:
                                    need_state = True

                            # Drum-boundary guard: if near bounds and PPO pushes further outward, override
                            margin = float(getattr(self, "nmpc_shield_drum_margin", 0.02))
                            a_i = np.asarray(actions_2d[i], dtype=float).reshape(-1)  # env action in [-1,1]
                            if isinstance(obs_i, dict):
                                if "drum_angles" in obs_i:
                                    drums = np.asarray(obs_i["drum_angles"], dtype=float).reshape(-1)  # 0..1
                                elif "drum_angle" in obs_i:
                                    drums = np.asarray(obs_i["drum_angle"], dtype=float).reshape(-1)   # 0..1
                                else:
                                    drums = None
                                if drums is not None:
                                    # expand scalar to match action dim if needed
                                    if drums.size == 1 and a_i.size > 1:
                                        drums = np.repeat(drums, a_i.size)
                                    # Only check aligned dims
                                    m = min(drums.size, a_i.size)
                                    drums = drums[:m]
                                    a_i = a_i[:m]
                                    near_low = drums <= margin
                                    near_high = drums >= (1.0 - margin)
                                    pushing_down = a_i < 0.0
                                    pushing_up = a_i > 0.0
                                    if np.any(near_low & pushing_down) or np.any(near_high & pushing_up):
                                        need_state = True
                        except Exception:
                            need_state = False

                    need = need_info if shield_kind == "info" else need_state if shield_kind == "state" else (need_info or need_state)

                    if not need:
                        new_actions.append(actions_2d[i].astype(np.float32))
                        continue

                a_nmpc = call_nmpc(obs_i_from(obs, i), t_i, info_i)
                a_nmpc = np.asarray(a_nmpc, dtype=np.float32).reshape(act_dim)
                new_actions.append(a_nmpc)

            a_nmpc_all = np.stack(new_actions, axis=0)

            
            if mode == "mix":
                
                mixed = alpha * a_nmpc_all + (1.0 - alpha) * actions_2d.astype(np.float32)
                return mixed if actions.ndim > 1 else mixed.reshape(-1)

            return a_nmpc_all if actions.ndim > 1 else a_nmpc_all.reshape(-1)

        return actions

        #kamal_end


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                actions_tensor, values, log_probs = self.policy(obs_tensor)
                actions = actions_tensor.detach().cpu().numpy()


            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # kamal_start
            infos_prev = getattr(self, "_last_infos", None)

            # calling the _maybe_override_with_nmpc() to override or intercept actions in ENV space (same shape as clipped_actions)
            overridden_env_actions = self._maybe_override_with_nmpc(
                obs=self._last_obs,
                actions=clipped_actions,
                infos_prev=infos_prev,
            )

            # --- hybrid override stats (for InterventionLogger) ---
            # Track how often NMPC/hybrid changes the PPO-proposed env action.
            if not hasattr(self, "_nmpc_override_env_steps"):
                self._nmpc_override_env_steps = 0
            if not hasattr(self, "_nmpc_total_env_steps"):
                self._nmpc_total_env_steps = 0

            # Count one decision per parallel env at this step because SB3 can run multiple envs in parallel
            self._nmpc_total_env_steps += env.num_envs

            # Compare PPO env action (before override) vs overridden env action
            ppo_env_actions = np.asarray(clipped_actions, dtype=np.float32) # actions that PPO wanted to take
            ov_env_actions = np.asarray(overridden_env_actions, dtype=np.float32) # these are the actual executed actions in the environment depending on the mode.


            # ensure proper shape  [n_envs , action_dim]
            if ppo_env_actions.ndim == 1:
                ppo_env_actions = ppo_env_actions.reshape(-1, 1)
            if ov_env_actions.ndim == 1:
                ov_env_actions = ov_env_actions.reshape(-1, 1)

            diff_mask = np.any(np.abs(ov_env_actions - ppo_env_actions) > 1e-12, axis=1) # this is to detect or check if NMPC action is different from PPO action; if yes, then override occured
            self._nmpc_override_env_steps += int(diff_mask.sum()) # counts how many 'true' overrides and accumualate overtime. Also, each entry corresponds to an environment.

            # Persist per-step override mask for debugging/logging if needed
            self.override_mask = diff_mask

            # Apply NMPC/hybrid override to the action sent to the environment
            clipped_actions = overridden_env_actions

            # Credit handover:  if NMPC overrode, store the executed action in the rollout buffer so that PPO learn from the action that was actually executed. 
            #If override happened, replace the stored PPO action with the actually executed action, and recompute the log-probability of that executed action under the current PPO policy.
            
            if np.any(diff_mask): # if the environment had its action changed, then fix what will be stored in the buffer. 
                # Convert executed env action to policy action space (needed when using squashed Gaussian)
                # this block is necessary because in SB3 PPO may sample actions internally in policy space but env receives actions in environment space.
                # So when squash-output = True, PPO often works by mapping actions policy to environment actions using tanh
                # this ensures that env sees bounded actions but PPO stores and evaluates log-probabilties in the policy/transfomed space. 
                # so the line below prevent storing NMPC actions directly as if they were policy outputs when squashing is active.Therefore, the line maps the executed env action back into the 
                # policy-compatible action representation. This makes the srored action and calculated log-probs to be consistent
                if isinstance(self.action_space, spaces.Box) and self.policy.squash_output:
                    executed_policy_actions = self.policy.scale_action(ov_env_actions) #scale-action is used here bcos actions were actually executed in the env space. 
                else:                                                                  # But rollout buffer log-prob calculations expects actions in policy action space
                    executed_policy_actions = ov_env_actions

                # Update the numpy action that will be stored into the rollout buffer (policy space)
                if actions.ndim == 1:
                    actions = actions.reshape(-1, 1)
                actions[diff_mask] = executed_policy_actions[diff_mask]

                # Recompute log-prob under the current policy for the executed action
                # IMPORTANT: store log_probs detached (no grad) for RolloutBuffer.add()
                with th.no_grad():
                    executed_actions_th = th.as_tensor(actions, device=self.device) # convert the actions in numpy array to tensor 
                    distribution = self.policy.get_distribution(obs_tensor) # obtain the action distribution at current observation 
                    log_probs = distribution.log_prob(executed_actions_th)
                    if log_probs.dim() > 1:
                        log_probs = log_probs.sum(dim=1)

            # kamal_end



            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # kamal-start

            # --- DEBUG: check what PPO receives during NMPC warmstart/ just for sanity check. Can disable this to reduce flooding the terminal---
            if not hasattr(self, "_dbg_warmstart_print_ctr"):
                self._dbg_warmstart_print_ctr = 0

            self._dbg_warmstart_print_ctr += 1

            # only print occasionally to avoid flooding logs
            PRINT_EVERY = 200

            override_mask = getattr(self, "override_mask", None)

            try:
        
                if self._dbg_warmstart_print_ctr % PRINT_EVERY == 0:
                    for i in range(len(infos)):
                        info_i = infos[i] if infos is not None else {}
                        cur_p = info_i.get("current_power", None)
                        des_p = info_i.get("desired_power", None)
                        perr  = info_i.get("power_error", None)

                        ov = None
                        if override_mask is not None:
                            try:
                                ov = bool(override_mask[i].item())
                            except Exception:
                                ov = None
                        mode_now = getattr(self, "nmpc_mode", "off")
                        print(
                        f"[HYBRID DBG][mode={mode_now}] t={self.num_timesteps:>7d} env={i} "
                        f"override={ov} reward={float(rewards[i]): .3f} done={bool(dones[i])} "
                        f"current_power={cur_p} desired_power={des_p} power_error={perr}"
                        )
                        #print(
                            #f"[WARMSTART DBG] t={self.num_timesteps:>7d} env={i} "
                            #f"override={ov} reward={float(rewards[i]): .3f} done={bool(dones[i])} "
                            #f"current_power={cur_p} desired_power={des_p} power_error={perr}"
                        #)
            except Exception as e:
                print(f"[WARMSTART DBG] print failed: {e}")
            # --- end debug ---
            #Kamal_end

            # kamal_start: persist infos so NMPC can use time/flags on next step
            
            self._last_infos = infos

            # kamal_end




            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            # Kamal_start
            """
            In the original SB3 PPO implementation, log_probs do not need explicit detachment 
            because they are computed under torch.no_grad() during rollout collection.
            In the hybrid RL-NMPC modification, log_probs.detach() was added after recomputing 
            log-probabilities for overridden actions as an extra safeguard to ensure 
            the stored rollout log-probs remain fixed data.
            """

            # Detach log_probs before storing them in the rollout buffer.
            # PPO needs these as fixed "old policy" log-probabilities during training.
            # They must be treated as constants, not as tensors still connected to the
            # current computation graph; otherwise gradients could incorrectly flow
            # through the stored old log-probs and break the intended PPO ratio update.


            #log_probs = log_probs.detach()

            # Kamal_end 

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
