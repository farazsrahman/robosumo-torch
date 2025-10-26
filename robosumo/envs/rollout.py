import gymnasium as gym
import os
import torch
from dataclasses import dataclass, field

import numpy as np
import imageio

import robosumo.envs

from robosumo.policy_zoo.policy import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, load_from_tf_params, load_lstm_from_tf_params

# ---- Default Constants ----
env_name = "RoboSumo-Ant-vs-Ant-v0"
policy_names = ("mlp", "mlp")
param_versions = (1, 1)
max_episodes = 3
record_video = False
seed = 42

POLICY_FUNC = {
    "mlp": MLPPolicy,
    "lstm": LSTMPolicy,
}

@dataclass
class EpisodeData:
    """Minimal dataclass for storing episode rollout data."""
    agent_name: str = ""
    action: list = field(default_factory=list)
    obs: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    total_reward: list = field(default_factory=list)
    done: list = field(default_factory=list)
    infos: list = field(default_factory=list)


def rollout(max_episodes, record_video, seed, debug=False):
    # Construct paths to parameters
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    params_dir = os.path.join(curr_dir, "../../robosumo/policy_zoo/assets")
    agent_names = [env_name.split('-')[1].lower(), env_name.split('-')[3].lower()]
    param_paths = []
    for a, p, v in zip(agent_names, policy_names, param_versions):
        param_paths.append(
            os.path.join(params_dir, a, p, "agent-params-v{}.npy".format(v))
        )

    # Auto-detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if debug:
        print("Using device: {}".format(device))

    # Create environment (disable checker for multi-agent)
    env = gym.make(env_name, disable_env_checker=True)

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

    assert len(policy) == 2, "Code only tested for 2 policies..."

    # Load policy parameters
    for i in range(len(policy)):
        params = load_params(param_paths[i])
        
        # Use appropriate loader based on policy type
        if policy_names[i] == "mlp":
            load_from_tf_params(policy[i], params)
        elif policy_names[i] == "lstm":
            load_lstm_from_tf_params(policy[i], params)
        
        if debug:
            print("Loaded parameters for policy {} from {}".format(i, param_paths[i]))

    # Seed environment and libraries for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Play matches between the agents
    num_episodes, nstep = 0, 0
    total_reward = [0.0 for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    
    # Reset environment (gymnasium returns obs, info)
    observation, info = env.reset(seed=seed)
    
    # Video recording for all episodes
    frames = [] 

    # Create rollout lists for both policies and provide an initial episode
    # Use list comprehension to avoid shallow copy bug with * operator
    rollouts = [[] for _ in range(len(policy))]
    for i in range(len(policy)):
        rollouts[i].append(EpisodeData())
        rollouts[i][-1].agent_name = agent_names[i]

    if debug:
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
        new_obs, reward, terminated, truncated, infos = env.step(action)
        # For multi-agent, terminated/truncated are already lists
        done = [t or tr for t, tr in zip(terminated, truncated)]

        # TODO (faraz): move this into the Env code
        # If any agent has 'winner': True in infos, set 'loser': True for all others
        winner_indices = [i for i, info in enumerate(infos) if info.get('winner') is True]
        if winner_indices:
            for i, info in enumerate(infos):
                if i not in winner_indices:
                    info['loser'] = True
        else: 
            for i, info in enumerate(infos):
                info['loser']  = False
                info['winner'] = False

        nstep += 1
        for i in range(len(policy)):
            total_reward[i] += reward[i]
            rollouts[i][-1].action.append(action[i])
            rollouts[i][-1].obs.append(observation[i])
            rollouts[i][-1].reward.append(reward[i])
            rollouts[i][-1].total_reward.append(total_reward[i])
            rollouts[i][-1].done.append(done[i])
            rollouts[i][-1].infos.append(infos[i]) 

        observation = new_obs # this is so that the action is paired with the observation that induced it and the reward that resulted from it 

        if done[0]:
            num_episodes += 1
            if debug:
                print("Episode {} finished after {} steps.".format(num_episodes, nstep))
            
            draw = True
            for i in range(len(policy)):
                if 'winner' in infos[i]:
                    draw = False
                    total_scores[i] += 1
                    if debug:
                        print("Winner: Agent {}, Scores: {}, Total Episodes: {}"
                              .format(i, total_scores, num_episodes))
            if draw and debug:
                print("Match tied: Agent {}, Scores: {}, Total Episodes: {}"
                      .format(i, total_scores, num_episodes))
            
            # Save video after each episode
            if record_video and len(frames) > 0:
                out_dir = "out"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                video_path = os.path.join(out_dir, "robosumo_episode{}.mp4".format(num_episodes))
                if debug:
                    print("Saving video to {}...".format(video_path))
                imageio.mimsave(video_path, frames, fps=30)
                if debug:
                    print("Video saved successfully!")
                frames = []  # Clear frames to free memory
            
            # Reset environment (gymnasium returns obs, info)
            observation, info = env.reset(seed=seed)
            nstep = 0
            total_reward = [0.0 for _ in range(len(policy))]

            for i in range(len(policy)):
                policy[i].reset()

            if num_episodes < max_episodes:
                if debug:
                    print("-" * 5 + "Episode {} ".format(num_episodes + 1) + "-" * 5)
                for i in range(len(policy)):
                    rollouts[i].append(EpisodeData())
                    rollouts[i][-1].agent_name = agent_names[i]

    return rollouts

def print_info(episodes: list[EpisodeData], agent_idx=0):
    """Prints info (steps, scores, draws) for a list of EpisodeData.
    
    Args:
        episodes: List of EpisodeData objects
        agent_idx: Which agent's perspective to print from (default: 0)
    """
    total_scores = 0
    n_episodes = len(episodes)
    
    for ep_num, ep in enumerate(episodes):
        print("-" * 5 + f"Episode {ep_num + 1} " + "-" * 5)
        n_steps = len(ep.action)
        print(f"Episode {ep_num + 1} finished after {n_steps} steps.")

        # Winner/loser/draw logic: check the infos from the last step
        if not ep.infos:
            print(f"Draw: Score: [{total_scores}, ...], Total Episodes: {ep_num + 1}")
            continue
            
        last_info = ep.infos[-1]
        is_winner = last_info.get('winner', False)
        is_loser = last_info.get('loser', False)

        # Determine and print outcome
        if is_winner:
            total_scores += 1
            print(f"Winner: Agent {agent_idx}, Score: [{total_scores}, ...], Total Episodes: {ep_num + 1}")
        elif is_loser:
            print(f"Loser: Agent {agent_idx}, Score: [{total_scores}, ...], Total Episodes: {ep_num + 1}")
        else:  # draw = if both winner AND loser are always False
            print(f"Match tied: Agent {agent_idx}, Score: [{total_scores}, ...], Total Episodes: {ep_num + 1}")

if __name__ == "__main__":
    rollouts = rollout()
    # Print info for first agent's episodes
    print_info(rollouts[0])