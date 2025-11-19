import gymnasium as gym
import os
import time
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

# Video fast mode settings (for faster rendering via frame rate downsampling)
VIDEO_FAST_MODE_DOWNSAMPLE = 5  # Capture every Nth frame (1 in 5 = 6 FPS when base is 30 FPS)
VIDEO_FAST_MODE_FPS = 30 // VIDEO_FAST_MODE_DOWNSAMPLE  # Effective FPS after downsampling

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
    log_prob: list = field(default_factory=list)

class EpisodeDataTorchMiniBatchIterator:
    """
    Converts a list of EpisodeData objects into flattened tensors suitable for PPO,
    and provides a mini-batch iterator that respects episode structure for GAE (i.e., no cross-episode mixing inside a batch).
    """

    def __init__(self, episodes: list[EpisodeData], device="cpu", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self._build_buffers(episodes)

    def _build_buffers(self, episodes: list[EpisodeData]):
        obs_list = []
        act_list = []
        reward_list = []
        done_list = []
        value_list = []
        log_prob_list = []
        episode_starts = []

        total_steps = 0

        for ep in episodes:
            if len(ep.obs) == 0:
                continue

            obs = torch.from_numpy(np.asarray(ep.obs, dtype=np.float32))
            actions = torch.from_numpy(np.asarray(ep.action, dtype=np.float32))
            rewards = torch.from_numpy(np.asarray(ep.reward, dtype=np.float32))
            dones = torch.from_numpy(np.asarray(ep.done, dtype=np.bool_))
            values = torch.from_numpy(np.asarray(ep.value, dtype=np.float32))
            log_probs = torch.from_numpy(np.asarray(ep.log_prob, dtype=np.float32))

            # Track episode start index
            start_idx = total_steps
            end_idx = start_idx + obs.shape[0]
            episode_starts.append(torch.arange(start_idx, end_idx, device=self.device, dtype=torch.long))

            obs_list.append(obs)
            act_list.append(actions)
            reward_list.append(rewards)
            done_list.append(dones)
            value_list.append(values)
            log_prob_list.append(log_probs)

            total_steps += obs.shape[0]

        if total_steps == 0:
            self.obs = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.actions = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.rewards = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.dones = torch.empty((0,), device=self.device, dtype=torch.bool)
            self.values = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.log_probs = torch.empty((0,), device=self.device, dtype=self.dtype)
            self.episode_starts = []
            return

        self.obs = torch.cat(obs_list, dim=0).to(self.device, dtype=self.dtype)
        self.actions = torch.cat(act_list, dim=0).to(self.device, dtype=self.dtype)
        self.rewards = torch.cat(reward_list, dim=0).to(self.device, dtype=self.dtype)
        self.dones = torch.cat(done_list, dim=0).to(self.device)
        self.values = torch.cat(value_list, dim=0).to(self.device, dtype=self.dtype)
        self.log_probs = torch.cat(log_prob_list, dim=0).to(self.device, dtype=self.dtype)
        self.episode_starts = episode_starts

    def __len__(self):
        return len(self.obs)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        
        This method handles multiple episodes correctly by using the done flags
        to reset bootstrap values and GAE traces at episode boundaries.
        """
        T = len(self.obs)
        returns = torch.zeros(T, device=self.device, dtype=self.dtype)
        advantages = torch.zeros(T, device=self.device, dtype=self.dtype)

        next_value = 0.0
        last_gae = 0.0

        for t in reversed(range(T)):
            non_terminal = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.values[t]
            # Set next_value for the previous timestep (in reverse order)
            # The non_terminal flag in the delta calculation above already handles
            # preventing bootstrapping at terminal states correctly
            next_value = self.values[t]

        self.returns = returns
        self.advantages = advantages
        return returns, advantages

    def iterate_minibatches(self, batch_size: int, shuffle=True):
        T = len(self.obs)
        if T == 0:
            return

        if shuffle:
            permutation = torch.randperm(T, device=self.device)
        else:
            permutation = torch.arange(T, device=self.device)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            indices = permutation[start:end]

            yield (
                self.obs[indices],
                self.actions[indices],
                self.returns[indices],
                self.advantages[indices],
                self.values[indices],
                self.log_probs[indices],
            )


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

def rollout(policy, env, seeds, record_video=False, debug=False, video_fast_mode=False, video_dir="out"):
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
    render_times = []  # Track render call times for profiling
    frame_counter = 0  # For frame downsampling in fast mode

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
        
        # Run inference with no gradient tracking
        with torch.no_grad():
            values = [
                pi.value(observation[i])
                for i, pi in enumerate(policy)
            ]
            policy_outputs = [
                pi.act(observation[i], stochastic=True)
                for i, pi in enumerate(policy)
            ]
            action = tuple(output[0] for output in policy_outputs)
            policy_infos = [output[1] for output in policy_outputs]
        
        # Step environment (gymnasium returns 5 values)
        new_obs, reward, terminated, truncated, infos = env.step(action)
        
        # Capture frame for video AFTER step (to capture updated state)
        if record_video:
            frame_counter += 1
            should_capture = not video_fast_mode or (frame_counter % VIDEO_FAST_MODE_DOWNSAMPLE == 1)
            if should_capture:
                render_start = time.time()
                frame = env.render()
                render_time = time.time() - render_start
                render_times.append(render_time)
                if frame is not None:
                    frames.append(frame)
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
            rollouts[i][-1].log_prob.append(policy_infos[i]['log_prob'])

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
            
            # Print render profiling stats if debug mode
            if debug and record_video and render_times:
                total_render_time = sum(render_times)
                avg_render_time = total_render_time / len(render_times)
                max_render_time = max(render_times)
                min_render_time = min(render_times)
                print(f"Render profiling: {len(render_times)} renders, "
                      f"total: {total_render_time:.3f}s, "
                      f"avg: {avg_render_time*1000:.2f}ms, "
                      f"min: {min_render_time*1000:.2f}ms, "
                      f"max: {max_render_time*1000:.2f}ms")
            
            # Save outputs (videos and plots) after each episode
            episode_value_histories = [rollouts[idx][-1].value for idx in range(len(policy))]
            should_save_video = record_video and len(frames) > 0
            # should_save_plot = debug and bool(episode_value_histories[0])
            should_save_plot = False
            if should_save_video or should_save_plot:
                save_video_w_value(
                    episode_idx=num_episodes,
                    frames=list(frames) if should_save_video else [],
                    value_histories=episode_value_histories,
                    agent_labels=agent_labels,
                    out_dir=video_dir,
                    fps=VIDEO_FAST_MODE_FPS if video_fast_mode else 30,
                    debug=debug,
                    save_plot=should_save_plot,
                    profile=debug,  # Profile when debug is enabled
                )
            frames = []  # Clear frames to free memory
            render_times = []  # Reset render times for next episode
            frame_counter = 0  # Reset frame counter for next episode
            
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
                frame_counter = 0  # Reset frame counter for next episode

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