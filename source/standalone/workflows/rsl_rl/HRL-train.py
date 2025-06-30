#!/usr/bin/env python3

"""
Hierarchical Reinforcement Learning (HRL) Training Script
- High-level policy: Selects which pre-trained policy to use
- Low-level policy: Determines specific parameters (e.g., target points for navigation)
"""

import os
import time
import torch
import numpy as np
import argparse
from collections import deque
import statistics
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

class HierarchicalActorCritic(torch.nn.Module):
    """
    Hierarchical Actor-Critic model with high-level and low-level policies.
    - High-level policy: Selects which pre-trained policy to use
    - Low-level policy: Determines specific parameters (target points)
    """
    def __init__(self,
                 num_obs,
                 num_critic_obs,
                 num_actions,
                 num_policies,
                 high_level_hidden_dims=[128, 64],
                 low_level_hidden_dims=[256, 256, 256],
                 activation='elu',
                 init_noise_std=1.0):
        super(HierarchicalActorCritic, self).__init__()
        
        self.num_policies = num_policies
        self.num_actions = num_actions
        
        # High-level policy (policy selector)
        self.high_level_actor_critic = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_policies,  # Output is policy selection
            actor_hidden_dims=high_level_hidden_dims,
            critic_hidden_dims=high_level_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std
        )
        
        # Low-level policy (parameter controller)
        self.low_level_actor_critic = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,  # Actual action space
            actor_hidden_dims=low_level_hidden_dims,
            critic_hidden_dims=low_level_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std
        )
        
        # Pre-trained policies storage
        self.pre_trained_policies = []
        
        # Flags
        self.is_recurrent = False
        self.distribution = None
        
    def load_pre_trained_policies(self, policy_paths):
        """Load pre-trained policies from disk"""
        assert len(policy_paths) == self.num_policies, f"Expected {self.num_policies} policy paths, got {len(policy_paths)}"
        
        self.pre_trained_policies = []
        for path in policy_paths:
            policy = ActorCritic(
                num_actor_obs=self.low_level_actor_critic.actor[0].in_features,
                num_critic_obs=self.low_level_actor_critic.critic[0].in_features,
                num_actions=self.num_actions
            )
            checkpoint = torch.load(path, map_location=next(self.parameters()).device)
            policy.load_state_dict(checkpoint['model_state_dict'])
            policy.eval()  # Set to evaluation mode
            self.pre_trained_policies.append(policy)
        
        print(f"Loaded {len(self.pre_trained_policies)} pre-trained policies")
    
    def reset(self, dones=None):
        self.high_level_actor_critic.reset(dones)
        self.low_level_actor_critic.reset(dones)
    
    def act(self, obs):
        """
        Two-stage action selection:
        1. High-level policy selects which pre-trained policy to use
        2. Low-level policy determines specific parameters
        """
        # Get policy selection from high-level policy
        self.high_level_actor_critic.update_distribution(obs)
        policy_selection = self.high_level_actor_critic.distribution.sample()
        
        # Get target parameters from low-level policy
        self.low_level_actor_critic.update_distribution(obs)
        target_params = self.low_level_actor_critic.distribution.sample()
        
        # Store distributions for later use
        self.high_level_distribution = self.high_level_actor_critic.distribution
        self.low_level_distribution = self.low_level_actor_critic.distribution
        
        # Convert to integers and clamp to valid range
        policy_indices = torch.clamp(policy_selection.round().long(), 0, self.num_policies - 1)
        
        # Store for computing log probs later
        self.policy_indices = policy_indices
        self.target_params = target_params
        
        return policy_indices, target_params
    
    def act_inference(self, obs):
        """Deterministic action selection for inference"""
        policy_selection = self.high_level_actor_critic.act_inference(obs)
        target_params = self.low_level_actor_critic.act_inference(obs)
        
        policy_indices = torch.clamp(policy_selection.round().long(), 0, self.num_policies - 1)
        
        return policy_indices, target_params
    
    def evaluate(self, critic_obs):
        """Evaluate state value using both high and low level critics"""
        high_value = self.high_level_actor_critic.evaluate(critic_obs)
        low_value = self.low_level_actor_critic.evaluate(critic_obs)
        
        # Combine values (could use different strategies here)
        return (high_value + low_value) / 2.0
    
    def get_actions_log_prob(self, actions):
        """Get log probabilities of the hierarchical action"""
        policy_indices, target_params = actions
        
        high_log_prob = self.high_level_distribution.log_prob(policy_indices.float()).sum(dim=-1)
        low_log_prob = self.low_level_distribution.log_prob(target_params).sum(dim=-1)
        
        # Combine log probabilities (simple sum for now)
        return high_log_prob + low_log_prob
    
    @property
    def action_mean(self):
        """Get mean of the action distribution (used for logging)"""
        return self.low_level_distribution.mean
    
    @property
    def action_std(self):
        """Get std of the action distribution (used for logging)"""
        return self.low_level_distribution.stddev
    
    @property
    def entropy(self):
        """Get entropy of both distributions (used for loss calculation)"""
        high_entropy = self.high_level_distribution.entropy().sum(dim=-1)
        low_entropy = self.low_level_distribution.entropy().sum(dim=-1)
        return high_entropy + low_entropy


class HierarchicalPPO(PPO):
    """Extended PPO for hierarchical policies"""
    
    def __init__(self, *args, **kwargs):
        super(HierarchicalPPO, self).__init__(*args, **kwargs)
    
    def act(self, obs, critic_obs):
        """Override to handle hierarchical actions"""
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
            
        # Get hierarchical actions
        policy_indices, target_params = self.actor_critic.act(obs)
        
        # Store combined actions as a tuple
        self.transition.actions = (policy_indices.detach(), target_params.detach())
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        
        # Record observations
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        return self.transition.actions


class HRLEnvironmentWrapper:
    """
    Wrapper around environment to handle hierarchical actions
    - Translates hierarchical actions to environment actions
    - Uses pre-trained policies with specified parameters
    """
    def __init__(self, env, hierarchical_actor_critic):
        self.env = env
        self.hac = hierarchical_actor_critic
        self.num_envs = env.num_envs
        self.num_obs = env.num_obs
        self.num_privileged_obs = env.num_privileged_obs
        self.num_actions = env.num_actions
    
    def reset(self):
        return self.env.reset()
    
    def step(self, hierarchical_actions):
        """
        Process hierarchical actions:
        1. Unpack policy_indices and target_params
        2. For each environment:
           a. Select appropriate pre-trained policy
           b. Use that policy with target parameters to generate actual actions
        3. Send actions to environment
        """
        policy_indices, target_params = hierarchical_actions
        
        # Initialize action tensor
        actions = torch.zeros(
            (self.num_envs, self.env.num_actions), 
            device=policy_indices.device
        )
        
        # Get current observations
        obs = self.env.get_observations()
        
        # For each environment instance
        for i in range(self.num_envs):
            policy_idx = policy_indices[i].item()
            
            # Use appropriate pre-trained policy if available
            if 0 <= policy_idx < len(self.hac.pre_trained_policies):
                policy = self.hac.pre_trained_policies[policy_idx]
                
                # Generate actions using the selected policy
                with torch.no_grad():
                    # Combine observation with target parameters for context
                    policy_input = obs[i:i+1]
                    policy_actions = policy.act_inference(policy_input)
                    
                    # Apply target parameters (e.g., adjust direction toward target point)
                    # This is a simplified example - actual implementation depends on action space
                    target_direction = target_params[i]
                    actions[i] = policy_actions * target_direction.abs()
            else:
                # Fallback if policy index is invalid
                actions[i] = target_params[i]
        
        # Step the environment with the computed actions
        return self.env.step(actions)
    
    # Forward all other methods to the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)


class HierarchicalOnPolicyRunner(OnPolicyRunner):
    """Extended OnPolicyRunner for hierarchical RL"""
    
    def __init__(self, env, train_cfg, policy_paths=None, **kwargs):
        # Initialize parent without creating the policy
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = kwargs.get('device', 'cpu')
        self.env = env
        
        # Get observation and action dimensions
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
            
        # Create hierarchical actor-critic
        self.hierarchical_actor_critic = HierarchicalActorCritic(
            num_obs=self.env.num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            num_policies=train_cfg.get("num_policies", 3),
            **self.policy_cfg
        )
        
        # Load pre-trained policies if provided
        if policy_paths:
            self.hierarchical_actor_critic.load_pre_trained_policies(policy_paths)
            
        # Create the algorithm
        self.alg = HierarchicalPPO(
            self.hierarchical_actor_critic, 
            device=self.device, 
            **self.alg_cfg
        )
        
        # Wrap environment
        self.wrapped_env = HRLEnvironmentWrapper(env, self.hierarchical_actor_critic)
        
        # Setup storage and other parameters
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # Initialize storage - note the action shape is different for hierarchical actions
        action_shape = [self.env.num_policies, self.env.num_actions]
        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [num_critic_obs], 
            action_shape
        )
        
        # Logging
        self.log_dir = kwargs.get('log_dir')
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        # Reset environment
        self.wrapped_env.reset()


def main():
    """Main function to train hierarchical RL agent"""
    parser = argparse.ArgumentParser(description="Train a hierarchical RL agent")
    parser.add_argument("--task", type=str, required=True, help="Task to train on")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to train on")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--num_policies", type=int, default=3, help="Number of pre-trained policies")
    parser.add_argument("--policy_dir", type=str, required=True, help="Directory containing pre-trained policies")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment
    env = gym.make(args.task, num_envs=args.num_envs)
    
    # Load pre-trained policy paths
    policy_paths = []
    for i in range(args.num_policies):
        path = os.path.join(args.policy_dir, f"policy_{i}.pt")
        if os.path.exists(path):
            policy_paths.append(path)
        else:
            print(f"Warning: Policy file {path} not found")
    
    # Create configuration dictionary
    train_cfg = {
        "runner": {
            "policy_class_name": "HierarchicalActorCritic",
            "algorithm_class_name": "HierarchicalPPO",
            "num_steps_per_env": 24,
            "save_interval": 50
        },
        "policy": {
            "high_level_hidden_dims": [128, 64],
            "low_level_hidden_dims": [256, 256, 256],
            "activation": "elu",
            "init_noise_std": 1.0
        },
        "algorithm": {
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.01,
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "max_grad_norm": 1.0
        },
        "num_policies": args.num_policies
    }
    
    # Create log directory
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, f"hrl_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create runner
    runner = HierarchicalOnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        policy_paths=policy_paths,
        log_dir=log_dir,
        device=args.device
    )
    
    # Train
    num_iterations = 500
    runner.learn(num_iterations)
    
    # Save final model
    runner.save(os.path.join(log_dir, f"model_final.pt"))
    
    print(f"Training completed! Model saved to {log_dir}")


if __name__ == "__main__":
    main()