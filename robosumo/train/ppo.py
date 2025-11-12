import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from robosumo.envs.rollout import (
    EpisodeDataTorchMiniBatchIterator,
    get_agents_and_env,
    rollout,
)
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
    total_updates=2,
    steps_per_update=512,
    lr=3e-4,
    val_freq=10_000_000,
    train_actor=True,
    train_critic=True,
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
    optimizer = optim.Adam(agent_policy.parameters(), lr=lr)


    for update_idx in range(total_updates):
        print(f"=== PPO Update {update_idx + 1}/{total_updates} ===")

        # 1) Collect trajectories by rolling out the current policies
        #    Note: we run multiple episodes and then stitch them together.
        record_validation_video = (update_idx + 1) % val_freq == 0

        with torch.no_grad():
            # Use fixed seeds per episode to keep behavior stable between updates
            # seeds = [1, 2, 3] 
            seeds = [np.random.randint(1, 1000000) for _ in range(3)] # HACK 
            episodes = rollout(
                policy=[agent_policy, frozen_policy],
                env=env,
                seeds=seeds if not record_validation_video else [42, 23, 21, 12], # HACK-y override to just get 1 video for speed
                record_video=record_validation_video,
                debug=False,
            )

        print(f"Episode 1 steps: {len(episodes[0][0].action)}")
        if record_validation_video:
            print(f"Saved validation rollout video at update {update_idx + 1}.")

        # 2) Flatten and prepare training data
        iterator = EpisodeDataTorchMiniBatchIterator(
            episodes=episodes[0], # HACK (Faraz): in the end we should have this take in multiple episodes, but I need to fix later
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

        if train_actor:
            advantages.copy_(normalize(advantages))

        # 3) PPO update epochs/minibatches
        if not train_actor and not train_critic:
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

                if train_actor:
                    if mb_log_probs is None:
                        raise RuntimeError("Old log probabilities missing for PPO actor update.")
                    ratio = torch.exp(new_log_probs - mb_log_probs)
                    surrogate = ratio * mb_advantages
                    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb_advantages
                    policy_loss = -torch.mean(torch.min(surrogate, clipped))
                    entropy_bonus = entropy_coef * pd.entropy().mean()
                    loss = loss + policy_loss - entropy_bonus

                if train_critic:
                    value_pred = value_pred.squeeze(-1)
                    value_clipped = mb_values + torch.clamp(value_pred - mb_values, -clip_epsilon, clip_epsilon)
                    value_loss_unclipped = (value_pred - mb_returns) ** 2
                    value_loss_clipped = (value_clipped - mb_returns) ** 2
                    value_loss = torch.mean(torch.max(value_loss_unclipped, value_loss_clipped))
                    loss = loss + value_coef * value_loss

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
                print(f"Stopped early due to reaching target KL ({last_kl:.4f}).")
                break

        print("Collected episodes and performed PPO update.\n")


if __name__ == "__main__":
    # Create agents and environment, then run the pseudo PPO loop
    policy_list, env = get_agents_and_env(
        debug=True,
        load_actor=[True, True],
        load_critic=[True, True],
    )

    # Expecting two agents; we'll train the first and keep the second frozen
    trainable_agent = policy_list[0]
    frozen_opponent = policy_list[1]

    # Ensure both are in eval mode for inference; training logic would switch as needed
    trainable_agent.train()  # allow updates to its parameters
    frozen_opponent.eval()   # keep opponent fixed

    run_ppo(trainable_agent, frozen_opponent, env, total_updates=1000, val_freq=500, train_actor=True, train_critic=True)