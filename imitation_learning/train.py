import os
import sys
import numpy as np
import h5py

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, export_policy_as_jit
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- CONFIGURATION ---
H5_FILE_PATH = "data/recorded_demo1.hdf5"
# Your exact XML path:
ROBOT_XML_PATH = "/home/koh-wh/Downloads/isaacgym_/unitree_rl_gym/resources/robots/g1_description/g1_29dof.xml"

BATCH_SIZE = 512
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
EXPECTED_OBS_DIM = 48 
# ---------------------

def train_from_recording(args):
    # 1. LOAD CONFIGURATION
    print("Loading configuration...")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # --- FORCE XML MODEL & JOINTS ---
    print(f"Overriding robot model with: {ROBOT_XML_PATH}")
    env_cfg.asset.file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof.xml" 
    # Note: We usually use the formatted string above if the file is inside resources, 
    # but since you have a specific absolute path, we can try setting it directly:
    env_cfg.asset.file = ROBOT_XML_PATH
    
    # CRITICAL FIX: Force the config to expect 29 joints
    # This ensures the network output layer is size 29, not 12
    env_cfg.env.num_actions = 29
    
    # 2. INITIALIZE ENVIRONMENT
    print("Initializing environment to build network architecture...")
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    actor = runner.alg.actor_critic.actor
    actor.to(env.device)
    actor.train()

    # Verify Network Output
    network_output_dim = 0
    for module in reversed(actor):
        if isinstance(module, nn.Linear):
            network_output_dim = module.out_features
            break
    print(f"Network Architecture Built. Expects {network_output_dim} actions.")

    # 3. LOAD DATASET
    print(f"Loading recording from: {H5_FILE_PATH}")
    obs_list = []
    act_list = []

    try:
        with h5py.File(H5_FILE_PATH, 'r') as f:
            data_group = f['data']
            for demo_name in data_group.keys():
                demo = data_group[demo_name]
                
                # --- EXTRACT RAW DATA ---
                joint_pos = torch.from_numpy(demo['obs']['robot_joint_pos'][:]).float().to(env.device)
                root_pos  = torch.from_numpy(demo['obs']['robot_root_pos'][:]).float().to(env.device)
                root_rot  = torch.from_numpy(demo['obs']['robot_root_rot'][:]).float().to(env.device)
                
                # --- PREPARE OBSERVATIONS (INPUTS) ---
                current_obs = torch.cat([joint_pos, root_pos, root_rot], dim=1)
                
                # Pad to 48
                N, current_dim = current_obs.shape
                missing_dim = EXPECTED_OBS_DIM - current_dim
                
                if missing_dim > 0:
                    padding = torch.zeros((N, missing_dim), device=env.device)
                    final_obs = torch.cat([current_obs, padding], dim=1)
                else:
                    final_obs = current_obs[:, :EXPECTED_OBS_DIM]

                # --- PREPARE TARGETS (ACTIONS) ---
                # Check if recording (29) matches network (29)
                if joint_pos.shape[1] > network_output_dim:
                    targets = joint_pos[:, :network_output_dim]
                elif joint_pos.shape[1] < network_output_dim:
                    print(f"Warning: Network wants {network_output_dim} actions but file only has {joint_pos.shape[1]}. Padding targets.")
                    pad_act = torch.zeros((N, network_output_dim - joint_pos.shape[1]), device=env.device)
                    targets = torch.cat([joint_pos, pad_act], dim=1)
                else:
                    targets = joint_pos

                # Sync lengths
                min_len = min(final_obs.shape[0], targets.shape[0])
                obs_list.append(final_obs[:min_len])
                act_list.append(targets[:min_len])

        if not obs_list:
            print("Error: No data loaded.")
            return

        obs_data = torch.cat(obs_list, dim=0)
        act_data = torch.cat(act_list, dim=0)
        
        print(f"\nDataset Ready. Samples: {obs_data.shape[0]} | Input: {obs_data.shape[1]} | Output: {act_data.shape[1]}")

    except Exception as e:
        print(f"CRITICAL ERROR loading H5: {e}")
        return

    # 4. TRAIN LOOP
    dataset = TensorDataset(obs_data, act_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Starting Training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch_obs, batch_acts in dataloader:
            predicted_acts = actor(batch_obs)
            loss = criterion(predicted_acts, batch_acts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {(total_loss/num_batches):.6f}")

    # 5. EXPORT
    export_folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit(runner.alg.actor_critic, export_folder)
    print(f"\nPolicy exported to: {export_folder}")

if __name__ == '__main__':
    args = get_args()
    train_from_recording(args)