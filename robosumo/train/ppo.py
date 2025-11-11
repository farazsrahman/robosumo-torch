import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from robosumo.envs.rollout import get_agents_and_env, rollout


def run_ppo(
    agent_policy,
    frozen_policy,
    env,
    total_updates=2,
    steps_per_update=512,
    lr=3e-4,
    val_freq=10_000_000,
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

    The numbered pseudo-code steps below sketch how these hyperparameters are used in
    a typical PPO implementation.
    """
    # Example optimizer over the trainable agent's parameters
    optimizer = optim.Adam(agent_policy.parameters(), lr=lr)

    # Use fixed seeds per episode to keep behavior stable between updates
    seeds = [1, 2, 3]

    for update_idx in range(total_updates):
        print(f"=== PPO Update {update_idx + 1}/{total_updates} ===")

        # 1) Collect trajectories by rolling out the current policies
        #    Note: we run multiple episodes and then stitch them together.
        record_validation_video = (update_idx + 1) % val_freq == 0

        with torch.no_grad():
            episodes = rollout(
                policy=[agent_policy, frozen_policy],
                env=env,
                seeds=seeds,
                record_video=record_validation_video,
                debug=False,
            )

        if record_validation_video:
            print(f"Saved validation rollout video at update {update_idx + 1}.")

        # 2) Pseudo-code: flatten and prepare training data
        #    - observations, actions, rewards, dones, values, log_probs (needed for PPO)
        #    - reuse hyperparameters listed in the docstring for the actual implementation.
        #    Here we only show the structure; actual tensors and computations are omitted.
        # observations = concat([episode.obs for episode in episodes[0]])
        # actions = concat([episode.action for episode in episodes[0]])
        # rewards = concat([episode.reward for episode in episodes[0]])
        # dones = concat([episode.done for episode in episodes[0]])
        # values = concat([episode.value for episode in episodes[0]])
        # old_log_probs = agent_policy.log_prob(observations, actions)

        # 3) Pseudo-code: compute returns and (optionally) GAE advantages
        # gamma = 0.99
        # gae_lambda = 0.95
        # returns = compute_discounted_returns(rewards, dones, gamma)
        # advantages = compute_gae(rewards, values, dones, gamma, gae_lambda)
        # if advantage_normalization:
        #     advantages = normalize(advantages)

        # 4) Pseudo-code: PPO update epochs/minibatches
        # ppo_epochs = 4
        # minibatch_size = 64
        # clip_epsilon = 0.2
        # value_coef = 0.5
        # entropy_coef = 0.01
        # max_grad_norm = 0.5
        # target_kl = 0.015
        # for epoch in range(ppo_epochs):
        #     for mb in iterate_minibatches(observations, actions, returns, advantages, old_log_probs, minibatch_size):
        #         mean, log_std, value = agent_policy.forward(mb.observations)
        #         new_log_probs = compute_log_probs(mean, log_std, mb.actions)
        #         ratio = torch.exp(new_log_probs - mb.old_log_probs)
        #         surrogate = ratio * mb.advantages
        #         clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * mb.advantages
        #         policy_loss = -torch.mean(torch.min(surrogate, clipped))
        #         clipped_values = values + torch.clamp(value - mb.values, -clip_epsilon, clip_epsilon)
        #         value_loss = value_coef * torch.mean((clipped_values - mb.returns) ** 2)
        #         entropy_bonus = entropy_coef * compute_entropy(mean, log_std)
        #         loss = policy_loss + value_loss - entropy_bonus
        #         optimizer.zero_grad()
        #         loss.backward()
        #         torch.nn.utils.clip_grad_norm_(agent_policy.parameters(), max_grad_norm)
        #         optimizer.step()
        #         if target_kl and estimated_kl(new_log_probs, mb.old_log_probs) > target_kl:
        #             break

        # For this pseudo loop, just perform a no-op optimizer step to show progress.
        optimizer.zero_grad()
        for p in agent_policy.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        optimizer.step()

        print("Collected episodes and performed a placeholder PPO update.\n")


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

    run_ppo(trainable_agent, frozen_opponent, env, total_updates=2, val_freq=9999)