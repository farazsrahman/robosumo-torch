import torch
import torch.nn as nn
import torch.optim as optim
import os
import tempfile
import glob
import shutil

import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from robosumo.envs.rollout import (
    EpisodeDataTorchMiniBatchIterator,
    get_agents_and_env,
    rollout,
)
from robosumo.envs.parallel_rollout import parallel_rollout
from robosumo.policy_zoo.utils import DiagonalGaussian


def estimated_kl(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Estimate the KL divergence between new and old policy distributions.
    """
    return torch.mean(old_log_probs - new_log_probs)


def normalize(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize a tensor to zero mean and unit variance.
    """
    if tensor.numel() == 0:
        return tensor
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    return (tensor - mean) / (std + eps)


def run_ppo(
    agent_policy,
    frozen_policy,
    env,
    cfg: DictConfig
):
    """
    Very simple pseudo-code style PPO loop using `MLPPolicy` and `rollout` data.

    This function intentionally outlines the steps and prints progress rather than
    implementing a full, numerically correct PPO.

    Core PPO hyperparameters (standard defaults indicated in parentheses):
        - learning_rate / optimizer lr (3e-4)
        - steps_per_update / rollout horizon (2048 or env-dependent)
        - gamma: discount factor for returns (0.99)
        - gae_lambda: GAE trace decay (0.95)
        - ppo_epochs: number of passes over collected data (4)
        - minibatch_size: size of minibatches inside an update (64)
        - clip_epsilon: policy ratio clipping range (0.2)
        - value_coef: weight on value-function loss (0.5)
        - entropy_coef: weight on policy entropy bonus (0.01)
        - max_grad_norm: gradient clipping norm (0.5)
        - advantage_normalization: whether to standardize advantages (True)
        - target_kl / early stopping threshold (optional, ~0.01-0.02)
        - train_actor / train_critic: toggles for optimizing the respective heads

    The numbered pseudo-code steps below sketch how these hyperparameters are used in
    a typical PPO implementation.
    """
    # Example optimizer over the trainable agent's parameters
    optimizer = optim.Adam(agent_policy.parameters(), lr=cfg.lr)

    # Initialize WandB with config
    wandb.init(
        project="robosumo-torch",
        name=cfg.run_name if cfg.run_name else None,
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    for update_idx in range(cfg.total_updates):
        print(f"=== PPO Update {update_idx + 1}/{cfg.total_updates} ===")

        # 1) Collect trajectories by rolling out the current policies
        #    Note: we run multiple episodes and then stitch them together.
        record_validation_video = (update_idx + 1) % cfg.val_freq == 0

        # Self-play: load trainable_agent weights to frozen_opponent every validation step
        if cfg.self_play and record_validation_video: # HACK using record_validation_video as a proxy for how often the opponent should be updated.
            frozen_policy.load_state_dict(agent_policy.state_dict())
            frozen_policy.eval()  # Ensure frozen policy stays in eval mode
            print(f"Loaded trainable_agent weights to frozen_opponent at update {update_idx + 1} (validation step)")

        video_dir = None
        if record_validation_video:
            video_dir = tempfile.mkdtemp(prefix="robosumo_videos_")

        with torch.no_grad():
            # Use fixed seeds per episode to keep behavior stable between updates
            # seeds = [1, 2, 3]
            seed = lambda: [np.random.randint(1, 1000000) for _ in range(cfg.n_rollouts_per_worker)]
            seeds = [seed() for _ in range (cfg.n_rollout_workers)] # HACK 
            if True and record_validation_video:
                seeds = [67] # HACK-y override to just get 1 video for speed
                print(f"Overriding seeds with {seeds} for video recording")
            # episodes = rollout(
            #     policy=[agent_policy, frozen_policy],
            #     env=env,
            #     seeds=seeds, 
            #     record_video=record_validation_video,
            #     video_fast_mode=True,
            #     debug=False,
            #     video_dir=video_dir if record_validation_video else None
            # )
            episodes = parallel_rollout(
                policy=[agent_policy, frozen_policy],
                env=env,
                seeds=seeds, 
                record_video=record_validation_video,
                video_fast_mode=True,
                debug=False,
                video_dir=video_dir if record_validation_video else None
            )



        # episodes is a list of lists: episodes[agent_idx][episode_idx]
        # episodes[0] contains all episodes for the trainable agent (agent 0)
        trainable_agent_episodes = episodes[0]
        num_episodes_collected = len(trainable_agent_episodes)
        total_steps = sum(len(ep.action) for ep in trainable_agent_episodes)
        
        # Compute and log the average episode duration (length in steps) to wandb
        if num_episodes_collected > 0:
            avg_steps = total_steps / len(trainable_agent_episodes)
            wandb.log({"avg_episode_duration": avg_steps}, step=update_idx)

        print(f"Collected {num_episodes_collected} episode(s) with {total_steps} total steps")
        if num_episodes_collected > 0:
            # print(f"  Episode lengths: {[len(ep.action) for ep in trainable_agent_episodes]}")
            pass 
        if record_validation_video and video_dir:
            # Upload videos to WandB
            video_files = glob.glob(os.path.join(video_dir, "robosumo_episode*_values_video.mp4"))
            for video_path in sorted(video_files):
                # Extract episode number from filename like "robosumo_episode1_values_video.mp4"
                filename = os.path.basename(video_path)
                episode_num = filename.split("episode")[1].split("_")[0]
                wandb.log({f"video/episode_{episode_num}": wandb.Video(video_path, format="mp4")}, step=update_idx)
            
            # Clean up temp directory
            shutil.rmtree(video_dir)
            print(f"Uploaded validation videos to WandB and cleaned up temp directory at update {update_idx + 1}.")

        # 2) Flatten and prepare training data from all episodes
        iterator = EpisodeDataTorchMiniBatchIterator(
            episodes=trainable_agent_episodes,
            device=agent_policy.get_device(),
            dtype=torch.float32,
        )

        if len(iterator) == 0:
            print("No rollout data collected; skipping update.\n")
            continue

        returns, advantages = iterator.compute_returns_and_advantages(
            gamma=0.99,
            gae_lambda=0.95,
        )

        if cfg.train_actor:
            advantages.copy_(normalize(advantages))

        # 3) PPO update epochs/minibatches
        if not cfg.train_actor and not cfg.train_critic:
            print("Both actor and critic training disabled; skipping optimization step.\n")
            continue

        ppo_epochs = 4
        minibatch_size = 64
        clip_epsilon = 0.2
        value_coef = 0.5
        entropy_coef = 0.01
        max_grad_norm = 0.5
        target_kl = 0.015

        early_stop = False
        last_kl = None
        
        # Track losses for logging
        epoch_policy_losses = []
        epoch_value_losses = []

        for epoch in range(ppo_epochs):
            for (
                mb_obs,
                mb_actions,
                mb_returns,
                mb_advantages,
                mb_values,
                mb_log_probs,
            ) in iterator.iterate_minibatches(minibatch_size, shuffle=True):
                mean, log_std, value_pred, aux = agent_policy.forward(mb_obs)
                pd = aux['distribution']
                new_log_probs = pd.log_prob(mb_actions)

                loss = torch.tensor(0.0, device=agent_policy.get_device())

                if cfg.train_actor:
                    if mb_log_probs is None:
                        raise RuntimeError("Old log probabilities missing for PPO actor update.")
                    ratio = torch.exp(new_log_probs - mb_log_probs)
                    surrogate = ratio * mb_advantages
                    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                    policy_loss = -torch.mean(torch.min(surrogate, clipped))
                    entropy_bonus = entropy_coef * pd.entropy().mean()
                    loss = loss + policy_loss - entropy_bonus
                    epoch_policy_losses.append(policy_loss.item())

                if cfg.train_critic:
                    value_pred = value_pred.squeeze(-1)
                    value_clipped = mb_values + torch.clamp(value_pred - mb_values, -clip_epsilon, clip_epsilon)
                    value_loss_unclipped = (value_pred - mb_returns) ** 2
                    value_loss_clipped = (value_clipped - mb_returns) ** 2
                    value_loss = torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
                    loss = loss + value_coef * value_loss
                    epoch_value_losses.append(value_loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent_policy.parameters(), max_grad_norm)
                optimizer.step()

                if target_kl is not None:
                    approx_kl = estimated_kl(new_log_probs, mb_log_probs)
                    last_kl = approx_kl.item()
                    if approx_kl > target_kl:
                        early_stop = True
                        break
            if early_stop:
                print(f"Stopped early after {epoch + 1}/{ppo_epochs} epochs due to reaching target KL ({last_kl:.4f}).")
                break

        # Log losses to WandB
        if epoch_policy_losses:
            wandb.log({"policy_loss": np.mean(epoch_policy_losses)}, step=update_idx)
        if epoch_value_losses:
            wandb.log({"value_loss": np.mean(epoch_value_losses)}, step=update_idx)

        print("Collected episodes and performed PPO update.\n")


@hydra.main(version_base=None, config_path="../../configs", config_name="ppo_config")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print()
    
    # Create agents and environment, then run the pseudo PPO loop
    policy_list, env = get_agents_and_env(
        debug=True,
        load_actor=list(cfg.load_actor),
        load_critic=list(cfg.load_critic),
    )

    # Expecting two agents; we'll train the first and keep the second frozen
    trainable_agent = policy_list[0]
    frozen_opponent = policy_list[1]

    # Ensure both are in eval mode for inference; training logic would switch as needed
    trainable_agent.train()  # allow updates to its parameters
    frozen_opponent.eval()   # keep opponent fixed

    run_ppo(
        trainable_agent, 
        frozen_opponent, 
        env,
        cfg
    )


if __name__ == "__main__":
    main()