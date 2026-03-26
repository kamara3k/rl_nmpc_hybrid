This repository implements a hybrid control configuration that combines Nonlinear Model Predictive Control (NMPC) with Proximal Policy Optimization (PPO) 
to achieve safe, stable, and sample-efficient control of dynamic systems.
The framework leverages NMPC as an expert controller to guide reinforcement learning, reduce exploration and accelerate convergence toward optimal policies.
Key Features
    • Hybrid Control Modes:
        ◦ Warm-start: NMPC controls the system during early training to guide PPO
        ◦ Mix: integrate NMPC and PPO actions
        ◦ Shield: NMPC overrides PPO when safety constraints are violated
    • Reinforcement Learning
        ◦ Constraint-aware control using NMPC
        ◦ Reduced exploration of unsafe regions
    • Sample Efficiency Improvements
        ◦ Expert-guided learning
        ◦ Faster optimal policy convergence compared to pure PPO
**It is important to download editabe SB3 in your machine. Then replace the OnPolicyAlgorithm.py located in the common with the OnPolicyAlgorithm.py in this repo  
Repository Structure: the repo contains the follwoing scripts to conduct the simulation experiment

- main_hybrid.py        # Main training and evaluation script
- envs.py               # Reactor environments (single/multi)
- nmpc.py               # NMPC control algorithm that is used as the expert policy
- OnPolicyAlgorithm.py  # contains the may_be_override_with_nmpc(), whihc is used to intercept the ppo actions in the sb3 OnPolicyAlgorithm.py 
- requirements.txt      # Python dependencies
- environment.yml       # contains all the dependencies to create virtual environment to run the experiment
 
Installation
1. Clone the repository
git clone https://github.com/kamara3k/rl_nmpc_hybrid.git
cd rl-nmpc_hybrid
2. Create environment using Conda (recommended)
An environment.yml file is provided. Therefore, create the environment directly from it:
conda env create -f environment.yml
conda activate rl-nmpc
Note: The requirements.txt file is optional and mainly provided for users who prefer pip. The Conda environment is the recommended setup for this project.

Usage
Seed is set to None in the main_hybrid.py. Therefore, it is compulsory to set it before running the script for instance 'your chocie = 0 or 1 or 2'
Train PPO only
python main_hybrid.py --mode ppo --seed 'your choice'
Train hybrid (warm-start)
python main_hybrid.py --mode ppo --nmpc_mode warmstart --seed 'your choice'
Train hybrid (mix)
python main_hybrid.py --mode ppo --nmpc_mode mix 'your choice'
Train hybrid (shield)
python main_hybrid.py --mode ppo --nmpc_mode shield 'your choice'
Evaluate trained PPO
python main_hybrid.py --mode ppo_eval --model_path path_to_model
Run NMPC only
python main_hybrid.py --mode nmpc

Note: Performance may vary across random seeds due to the stochastic nature of PPO.

Dependencies
    • Python 3.10+
    • Stable-Baselines3
    • NumPy
    • SciPy
    • Matplotlib
    

If you use this work, please cite:
@article{nmpc_ppo_hybrid,
  title={Hybrid NMPC–PPO Control for Safe and Efficient Learning},
  author={Abdulraheem, Kamal},
  year={2025}
}License
This project is licensed under the MIT License.
Future Work
    • Improved credit assignment between NMPC and PPO
    • MPC-informed value function learning
    • Extension to multi-agent reactor systems
