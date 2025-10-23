"""
Demonstrates RoboSumo with pre-trained PyTorch policies.

This is the PyTorch port of the original play.py, using:
- Gymnasium API instead of Gym
- PyTorch policies instead of TensorFlow
- Modern MuJoCo bindings
- Direct policy inference
"""
import click
import gymnasium as gym
import os
import torch

import numpy as np
import imageio

import robosumo.envs

from robosumo.policy_zoo.policy import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, load_from_tf_params, load_lstm_from_tf_params


POLICY_FUNC = {
    "mlp": MLPPolicy,
    "lstm": LSTMPolicy,
}


@click.command()
@click.option("--env", type=str,
              default="RoboSumo-Ant-vs-Ant-v0", show_default=True,
              help="Name of the environment.")
@click.option("--policy-names", nargs=2, type=click.Choice(["mlp", "lstm"]),
              default=("mlp", "mlp"), show_default=True,
              help="Policy names.")
@click.option("--param-versions", nargs=2, type=int,
              default=(1, 1), show_default=True,
              help="Policy parameter versions.")
@click.option("--max_episodes", type=int,
              default=3, show_default=True,
              help="Number of episodes.")
@click.option("--record-video", is_flag=True, default=False, show_default=True,
              help="Record video for all episodes.")

def main(env, policy_names, param_versions, max_episodes, record_video):
    # Construct paths to parameters
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    params_dir = os.path.join(curr_dir, "robosumo/policy_zoo/assets")
    agent_names = [env.split('-')[1].lower(), env.split('-')[3].lower()]
    param_paths = []
    for a, p, v in zip(agent_names, policy_names, param_versions):
        param_paths.append(
            os.path.join(params_dir, a, p, "agent-params-v{}.npy".format(v))
        )

    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))

    # Create environment (disable checker for multi-agent)
    env = gym.make(env, disable_env_checker=True)

    # Adjust agent z-offset (same as original)
    # Access the unwrapped environment to modify agents
    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, 'agents'):
        for agent in unwrapped_env.agents:
            agent._adjust_z = -0.5

    # Initialize policies
    policy = []
    for i, name in enumerate(policy_names):
        policy.append(
            POLICY_FUNC[name](
                ob_space=env.observation_space.spaces[i],
                ac_space=env.action_space.spaces[i],
                hiddens=[64, 64],
                normalize=True,
                device=device
            )
        )
        # Set to evaluation mode
        policy[i].eval()

    # Load policy parameters
    for i in range(len(policy)):
        params = load_params(param_paths[i])
        
        # Use appropriate loader based on policy type
        if policy_names[i] == "mlp":
            load_from_tf_params(policy[i], params)
        elif policy_names[i] == "lstm":
            load_lstm_from_tf_params(policy[i], params)
        
        print("Loaded parameters for policy {} from {}".format(i, param_paths[i]))

    # Play matches between the agents
    num_episodes, nstep = 0, 0
    total_reward = [0.0 for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    
    # Reset environment (gymnasium returns obs, info)
    observation, info = env.reset()
    
    # Video recording for all episodes
    frames = [] 
    
    print("-" * 5 + "Episode {} ".format(num_episodes + 1) + "-" * 5)
    while num_episodes < max_episodes:
        
        # Capture frame for video
        if record_video:
            frame = env.render()
            frames.append(frame)
        
        # Run inference with no gradient tracking
        with torch.no_grad():
            action = tuple([
                pi.act(observation[i], stochastic=True)[0]
                for i, pi in enumerate(policy)
            ])
        
        # Step environment (gymnasium returns 5 values)
        observation, reward, terminated, truncated, infos = env.step(action)
        # For multi-agent, terminated/truncated are already lists
        done = [t or tr for t, tr in zip(terminated, truncated)]

        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
        
        if done[0]:
            # Print the number of iterations (steps) before the episode finished
            print("Episode {} finished after {} steps.".format(num_episodes + 1, nstep))
            num_episodes += 1
            draw = True
            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    print("Winner: Agent {}, Scores: {}, Total Episodes: {}"
                          .format(i, total_scores, num_episodes))
            if draw:
                print("Match tied: Agent {}, Scores: {}, Total Episodes: {}"
                      .format(i, total_scores, num_episodes))
            
            # Save video after each episode
            if record_video and len(frames) > 0:
                out_dir = "out"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                video_path = os.path.join(out_dir, "robosumo_episode{}.mp4".format(num_episodes))
                print("Saving video to {}...".format(video_path))
                imageio.mimsave(video_path, frames, fps=30)
                print("Video saved successfully!")
                frames = []  # Clear frames to free memory
            
            # Reset environment (gymnasium returns obs, info)
            observation, info = env.reset()
            nstep = 0
            total_reward = [0.0 for _ in range(len(policy))]

            for i in range(len(policy)):
                policy[i].reset()

            if num_episodes < max_episodes:
                print("-" * 5 + "Episode {} ".format(num_episodes + 1) + "-" * 5)


if __name__ == "__main__":
    main()

