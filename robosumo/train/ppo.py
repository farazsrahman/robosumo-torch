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
        #    Here we only show the structure; actual tensors and computations are omitted.
        # observations = ...
        # actions = ...
        # rewards = ...
        # dones = ...
        # values = ...
        # old_log_probs = ...

        # 3) Pseudo-code: compute returns and (optionally) GAE advantages
        # returns = compute_discounted_returns(rewards, dones, gamma)
        # advantages = compute_gae(rewards, values, dones, gamma, lam)

        # 4) Pseudo-code: PPO update epochs/minibatches
        # for epoch in range(ppo_epochs):
        #     for mb in minibatches(observations, actions, returns, advantages, old_log_probs):
        #         mean, log_std, value = agent_forward(observations_mb)
        #         new_log_probs = compute_log_probs(mean, log_std, actions_mb)
        #         ratio = exp(new_log_probs - old_log_probs_mb)
        #         policy_loss = -mean(min(ratio * adv_mb, clip(ratio, 1-eps, 1+eps) * adv_mb))
        #         value_loss = mse(value, returns_mb)
        #         entropy_bonus = entropy(mean, log_std)
        #         loss = policy_loss + c1 * value_loss - c2 * entropy_bonus
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

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
    policy_list, env = get_agents_and_env(debug=True)

    # Expecting two agents; we'll train the first and keep the second frozen
    trainable_agent = policy_list[0]
    frozen_opponent = policy_list[1]

    # Ensure both are in eval mode for inference; training logic would switch as needed
    trainable_agent.train()  # allow updates to its parameters
    frozen_opponent.eval()   # keep opponent fixed

    run_ppo(trainable_agent, frozen_opponent, env, total_updates=200, val_freq=100)