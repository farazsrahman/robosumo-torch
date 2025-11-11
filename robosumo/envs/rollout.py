import gymnasium as gym
import os
import torch
from dataclasses import dataclass, field

import numpy as np

import robosumo.envs

from robosumo.envs.vis import (
    get_agent_labels,
    save_video_w_value,
)
from robosumo.policy_zoo.policy import LSTMPolicy, MLPPolicy
from robosumo.policy_zoo.utils import load_params, load_from_tf_params, load_lstm_from_tf_params
# ---- Default Constants ----
env_name = "RoboSumo-Ant-vs-Ant-v0"
policy_names = ("mlp", "mlp")
param_versions = (1, 1)
record_video = True 
seeds = [10, 41, 43]
max_episodes = len(seeds)

POLICY_FUNC = {
    "mlp": MLPPolicy,
    "lstm": LSTMPolicy,
}

@dataclass
class EpisodeData:
    """Minimal dataclass for storing episode rollout data."""
    morphology: str = ""
    action: list = field(default_factory=list)
    obs: list = field(default_factory=list)
    reward: list = field(default_factory=list)
    total_reward: list = field(default_factory=list)
    done: list = field(default_factory=list)
    infos: list = field(default_factory=list)
    value: list = field(default_factory=list)


def set_seed(seed):
    """Set random seeds for numpy, torch, and cuda."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_agents_and_env(debug = False, load_actor=None, load_critic=None):
    
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
    # device = 'cpu'
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
    num_policies = len(policy_names)

    if load_actor is None:
        load_actor = [True] * num_policies
    if load_critic is None:
        load_critic = [True] * num_policies

    if len(load_actor) != num_policies or len(load_critic) != num_policies:
        raise ValueError("load_actor and load_critic must match the number of policies.")

    policy = []
    for i, name in enumerate(policy_names):
        policy.append(
            POLICY_FUNC[name](
                morphology=agent_names[i],
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
            if load_actor[i] or load_critic[i]:
                load_from_tf_params(
                    policy[i],
                    params,
                    load_actor=load_actor[i],
                    load_critic=load_critic[i],
                )
                if debug:
                    components = []
                    if load_actor[i]:
                        components.append("actor")
                    if load_critic[i]:
                        components.append("critic")
                    component_str = " & ".join(components)
                    print(f"Loaded {component_str} weights for policy {i} from {param_paths[i]}")
            elif debug:
                print(f"Skipped loading pretrained weights for policy {i}")
        elif policy_names[i] == "lstm":
            load_lstm_from_tf_params(policy[i], params)
        
        if debug and policy_names[i] != "mlp":
            print("Loaded parameters for policy {} from {}".format(i, param_paths[i]))

    return policy, env

def rollout(policy, env, seeds, record_video=False, debug=False):
    max_episodes = len(seeds)
    
    
    # Play matches between the agents
    num_episodes, nstep = 0, 0
    total_reward = [0.0 for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    
    # Seed environment and libraries with the first seed for initial setup
    seed = seeds[0]
    set_seed(seed)
    observation, info = env.reset(seed=seed)
    
    # Video recording for all episodes
    frames = []

    # Create rollout lists for both policies and provide an initial episode
    # Use list comprehension to avoid shallow copy bug with * operator
    rollouts = [[] for _ in range(len(policy))]
    for i in range(len(policy)):
        rollouts[i].append(EpisodeData())
        rollouts[i][-1].morphology = policy[i].morphology

    agent_labels = get_agent_labels(policy)

    if debug:
        print("-" * 5 + "Episode {} (seed: {}) ".format(num_episodes + 1, seeds[num_episodes]) + "-" * 5)
    
    while num_episodes < max_episodes:
        
        # Capture frame for video
        frame = None
        if record_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        # Run inference with no gradient tracking
        with torch.no_grad():
            values = [
                pi.value(observation[i])
                for i, pi in enumerate(policy)
            ]
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
            rollouts[i][-1].value.append(values[i])

        observation = new_obs # this is so that the action is paired with the observation that induced it and the reward that resulted from it 

        if done[0]:
            num_episodes += 1
            if debug:
                print("Episode {} finished after {} steps.".format(num_episodes, nstep))
            
            draw = True
            for i in range(len(policy)):
                if infos[i].get('winner') is True:
                    draw = False
                    total_scores[i] += 1
                    if debug:
                        print("Winner: Agent {}, Scores: {}, Total Episodes: {}"
                              .format(i, total_scores, num_episodes))
            if draw and debug:
                print("Match tied: Agent {}, Scores: {}, Total Episodes: {}"
                      .format(i, total_scores, num_episodes))
            
            # Save outputs (videos and plots) after each episode
            episode_value_histories = [rollouts[idx][-1].value for idx in range(len(policy))]
            should_save_video = record_video and len(frames) > 0
            should_save_plot = debug and bool(episode_value_histories[0])
            if should_save_video or should_save_plot:
                save_video_w_value(
                    episode_idx=num_episodes,
                    frames=list(frames) if should_save_video else [],
                    value_histories=episode_value_histories,
                    agent_labels=agent_labels,
                    out_dir="out",
                    fps=30,
                    debug=debug,
                    save_plot=should_save_plot,
                )
            frames = []  # Clear frames to free memory
            
            # Reset environment with next seed if there are more episodes
            if num_episodes < max_episodes:
                next_seed = seeds[num_episodes]
                set_seed(next_seed)
                observation, info = env.reset(seed=next_seed)
            nstep = 0
            total_reward = [0.0 for _ in range(len(policy))]

            for i in range(len(policy)):
                policy[i].reset()

            if num_episodes < max_episodes:
                if debug:
                    print("-" * 5 + "Episode {} (seed: {}) ".format(num_episodes + 1, seeds[num_episodes]) + "-" * 5)
                for i in range(len(policy)):
                    rollouts[i].append(EpisodeData())
                    rollouts[i][-1].morphology = policy[i].morphology
                frames = []

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
    import time

    policy, env = get_agents_and_env(debug=True)
    start_time = time.time()
    rollouts = rollout(policy=policy, env=env, seeds=seeds, record_video=record_video, debug=True)
    elapsed_time = time.time() - start_time
    print(f"Rollout execution time: {elapsed_time:.2f} seconds")
    # Print info for first agent's episodes
    print_info(rollouts[0])