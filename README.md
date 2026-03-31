# RL–NMPC Hybrid Control Configuration

## Overview

This repository implements a hybrid control configuration that combines **Nonlinear Model Predictive Control (NMPC)** with **Proximal Policy Optimization (PPO)** to achieve safe, stable, and sample-efficient control of a micro nuclear reactor system (HolosGen microreactor).

The framework leverages NMPC as an expert controller to guide reinforcement learning in other to reduce exploration, and accelerate convergence toward optimal policies.

---

## Key Features

* **Hybrid Control Modes**

  * **Warm-start**: NMPC controls the system during early training to guide PPO
  * **Mix**: Integrates NMPC and PPO actions
  * **Shield**: NMPC overrides PPO when safety constraints are violated

* **Reinforcement Learning**

  * Constraint-aware control using NMPC
  * Reduced exploration

* **Sample Efficiency Improvements**

  * Expert-guided learning
  * Faster convergence to optimal policy compared to pure PPO

---

## Important Setup Note

You must install an **editable version of Stable-Baselines3 (SB3)**.

Then replace the following file in the editable SB3 in your local machine with the OnPolicyAlgorithm.py in this repo:

```
<your_sb3_path>/stable_baselines3/common/OnPolicyAlgorithm.py
```

with the `OnPolicyAlgorithm.py` provided in this repository.

This modified file contains the function:

* `maybe_override_with_nmpc()`

which is used to intercept PPO actions and integrate NMPC within the training loop.

---

## Repository Structure

```
.
├── main_hybrid.py        # Main training and evaluation script
├── envs.py               # Reactor environments (single/multi)
├── nmpc.py               # NMPC control algorithm (expert policy)
├── on_policy_algorithm.py  # Modified SB3 file with NMPC override logic
├── requirements.txt      # Python dependencies (optional)
├── environment.yml       # Conda environment (recommended)
```

---
## Installation

### 1. Clone the repository
```bash
git clone https://github.com/kamara3k/rl_nmpc_hybrid.git
cd rl_nmpc_hybrid
```

### 2.
```bash
git clone git@github.com:DLR-RM/stable-baselines3.git
```

### 3.
```bash
cp on_policy_algorithm.py stable-baselines3/stable_baselines3/common/
```

### 4. Create environment (recommended)

An `environment.yml` file is provided. Create the environment using:

```bash
conda env create -f environment.yml
conda activate rl-nmpc
```
 **Note:** `requirements.txt` is optional and intended for users who prefer `pip`.

---

## Usage

 **Important:** The seed is set to `None` by default in `main_hybrid.py`. You must specify a seed when running experiments (e.g., 0, 1, or 2).

### Train PPO only

```bash
python main_hybrid.py --mode ppo --seed 0
```

### Train hybrid (warm-start)

```bash
python main_hybrid.py --mode ppo --nmpc_mode warmstart --seed 0
```

### Train hybrid (mix)

```bash
python main_hybrid.py --mode ppo --nmpc_mode mix --seed 0
```

### Train hybrid (shield)

```bash
python main_hybrid.py --mode ppo --nmpc_mode shield --seed 0
```

### Evaluate trained PPO

```bash
python main_hybrid.py --mode ppo_eval --model_path path_to_model
```

### Run NMPC only

```bash
python main_hybrid.py --mode nmpc
```

---

## Notes on Performance

* Performance may vary across random seeds due to the stochastic nature of PPO.
* Hybrid methods aim to improve safety and learning efficiency but may exhibit variability depending on configuration and seed.

---

## Dependencies

* Python 3.10+
* Stable-Baselines3
* NumPy
* SciPy
* Matplotlib

---
