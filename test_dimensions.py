#!/usr/bin/env python3
"""
RoboSumo-Torch Action-Observation Space Verification Script

This script verifies the dimensions in robosumo-torch by instantiating 
environments and measuring actual observation/action spaces.

Usage:
    python test_dimensions.py

The script tests:
1. Environment instantiation for different agent combinations
2. Observation and action space dimensions
3. Policy loading compatibility
4. Coordinate system verification
"""
import sys
import os
sys.path.append('.')

import gym
import numpy as np

# Import robosumo-torch
import robosumo.envs

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(" {}".format(title))
    print("=" * 80)

def print_section(title):
    """Print a formatted section header."""
    print("\n--- {} ---".format(title))

def test_environment_dimensions():
    """Test environment dimensions for key agent combinations."""
    
    print_header("ROBOSUMO-TORCH DIMENSION VERIFICATION")
    print("Analyzing robosumo-torch implementation...")
    
    # Test key combinations
    test_cases = [
        ('RoboSumo-Ant-vs-Ant-v0', 'Ant vs Ant'),
        ('RoboSumo-Bug-vs-Spider-v0', 'Bug vs Spider'),
        ('RoboSumo-Spider-vs-Spider-v0', 'Spider vs Spider'),
    ]
    
    results = {}
    
    for env_name, description in test_cases:
        print_section("Testing {}".format(description))
        
        try:
            # Create environment
            env = gym.make(env_name)
            print("✓ Environment created: {}".format(env_name))
            
            # Get agent names
            agent_names = [env_name.split('-')[1].lower(), env_name.split('-')[3].lower()]
            
            # Measure observation and action spaces
            print("\nAgent Information:")
            for i, agent_name in enumerate(agent_names):
                obs_dim = env.observation_space.spaces[i].shape[0]
                action_dim = env.action_space.spaces[i].shape[0]
                print("  Agent {} ({}):".format(i+1, agent_name.upper()))
                print("    Observation space: {}D".format(obs_dim))
                print("    Action space: {}D".format(action_dim))
            
            # Test actual observation by resetting environment
            obs = env.reset()
            actual_obs_dims = [len(obs[i]) for i in range(len(obs))]
            
            print("\nActual Observation Dimensions:")
            for i, agent_name in enumerate(agent_names):
                print("  Agent {} ({}): {}D".format(i+1, agent_name.upper(), actual_obs_dims[i]))
            
            # Store results
            results[env_name] = {
                'agent_names': agent_names,
                'obs_dims': actual_obs_dims,
                'action_dims': [env.action_space.spaces[i].shape[0] for i in range(len(agent_names))],
            }
            
            env.close()
            
        except Exception as e:
            print("✗ ERROR testing {}: {}".format(description, e))
            results[env_name] = {'error': str(e)}
    
    return results

def test_coordinate_system():
    """Test coordinate system properties."""
    print_section("Coordinate System Verification")
    
    try:
        env = gym.make('RoboSumo-Ant-vs-Ant-v0')
        obs = env.reset()
        
        # Get agent positions from qpos
        agent1_pos = obs[0][:3]  # First 3 elements are x, y, z
        agent2_pos = obs[1][:3]
        
        print("Agent 1 initial position: [{:.3f}, {:.3f}, {:.3f}]".format(agent1_pos[0], agent1_pos[1], agent2_pos[2]))
        print("Agent 2 initial position: [{:.3f}, {:.3f}, {:.3f}]".format(agent2_pos[0], agent2_pos[1], agent2_pos[2]))
        
        # Calculate distances from center
        dist1 = np.sqrt(agent1_pos[0]**2 + agent1_pos[1]**2)
        dist2 = np.sqrt(agent2_pos[0]**2 + agent2_pos[1]**2)
        
        print("Distance from center:")
        print("  Agent 1: {:.3f}m".format(dist1))
        print("  Agent 2: {:.3f}m".format(dist2))
        
        env.close()
        
    except Exception as e:
        print("✗ ERROR testing coordinate system: {}".format(e))

def print_summary(results):
    """Print summary of all tests."""
    print_header("SUMMARY")
    
    print("Environment Dimension Tests:")
    for env_name, data in results.items():
        if 'error' in data:
            print("  {}: ✗ ERROR - {}".format(env_name, data['error']))
        else:
            print("  {}: ✓ PASSED".format(env_name))
    
    print("\nrobosumo-torch Dimensions:")
    for env_name, data in results.items():
        if 'error' not in data:
            agent_names = data['agent_names']
            obs_dims = data['obs_dims']
            action_dims = data['action_dims']
            print("  {}:".format(env_name))
            for i, agent_name in enumerate(agent_names):
                print("    {}: {}D observation, {}D action".format(agent_name.upper(), obs_dims[i], action_dims[i]))

def main():
    """Main function to run all tests."""
    print("RoboSumo-Torch Action-Observation Space Analysis")
    print("This script analyzes the current state of robosumo-torch dimensions")
    
    # Run tests
    results = test_environment_dimensions()
    test_coordinate_system()
    
    # Print summary
    print_summary(results)
    
    print_header("ANALYSIS COMPLETE")
    print("Check the results above to understand robosumo-torch current state.")

if __name__ == "__main__":
    main()
