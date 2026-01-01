# Follow Setup commands in doc  
[setup.md](/doc/setup_en.md)  
[requirements.txt](/doc/requirements.txt)  
[notes.md](/doc/notes.md)  
  
# Folder Structure  
```
isaacgym_/
├── isaacgym/                
├── rsl_rl/   
├── unitree_rl_gym/              
├── unitree_sdk2_python/                    
```

Can drag xml files in window to view the different urdf models.  
```bash
python -m mujoco.viewer
```
  
Keyboard control of Unitree G1 Robot. Can use after setting up.
```bash
python keyboard_g1.py
```
  
# Reinforcement learning  
https://github.com/unitreerobotics/unitree_rl_gym  
[Readme(unitree).md](/doc/Readme(unitree).md)  
```bash
conda activate unitree-rl
```
```bash
cd Downloads/isaacgym_/unitree_rl_gym
```
  
## Train:
```bash
python legged_gym/scripts/train.py --task=g1 --headless --num_envs=512 --max_iterations=1000 --experiment_name=g1_walking_test --run_name=iterations1000
```
  
## Play:
```bash
python legged_gym/scripts/play.py --task=g1 --experiment_name=g1_walking_test --load_run={Run_name} --num_envs=32
```
  
## Sim-to-Sim:
Under '/Downloads/isaacgym_/unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml'  
Change policy path to new model and run:  
```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

### If above don't work do: 
--- 
Create the dri directory that MuJoCo is looking for & Link existing libraries to that location:  
```bash
sudo mkdir -p /usr/lib/dri  
sudo ln -sf /usr/lib/x86_64-linux-gnu/dri/* /usr/lib/dri/  
```
Check available GLIBCXX versions & Install updated libstdc++ in conda:  
```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
conda install -c conda-forge libstdcxx-ng
conda update libgcc
```
Run:  
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
``` 
Using this:( $__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia <app>)  
  
## Sim-to-Real:
  
## Other Behaviours:
- Go to $cd ~/Downloads/isaacgym_/unitree_rl_gym/legged_gym/envs/  
- Create new "config" and "env" scripts for particular action.  
- Include in "init.py"  
- Run Train and the rest.  

Example g1_squat:  
```bash
python legged_gym/scripts/train.py --task=g1_squatting --headless --num_envs=512 --max_iterations=2000 --experiment_name=g1_squatting --run_name=iterations
```
```bash
python legged_gym/scripts/play.py --task=g1_squatting --experiment_name=g1_squatting --load_run=it2000 --num_envs=32
```
```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```

Example g1_standing still:  
```bash
python legged_gym/scripts/train.py --task=g1_standing --headless --num_envs=512 --max_iterations=2000 --experiment_name=g1_standing --run_name=iterations
```
```bash
python legged_gym/scripts/play.py --task=g1_standing --experiment_name=g1_standing --load_run=it2000 --num_envs=32
```
```bash
python deploy/deploy_mujoco/deploy_mujoco.py g1.yaml
```
---  

## Install unitree-rl environment 
### Navigate to folder with the envrionment file 
conda env create -f environment.yml  
### 1. Install Isaac Gym
cd isaacgym/python  
pip install -e .  
### 2. Install RSL_RL (The reinforcement learning library)
cd ../../rsl_rl  
pip install -e .  
### 3. Install Unitree RL Gym (The robot-specific training code)
cd ../unitree_rl_gym  
pip install -e .  