
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
from robosumo.envs.rollout import get_agents_and_env, rollout, print_info 

import multiprocessing
from tqdm import tqdm
import copy

def single_rollout(policy, env, seeds):
    return rollout(
        policy=policy,
        env=env,
        seeds=seeds,
        record_video=False,
        debug=False
    )

def _worker_rollout(args):
    """Worker function for multiprocessing - must be at module level."""
    seeds, worker_policies = args
    
    # Each process needs its own environment (MuJoCo environments are not thread-safe)
    # But we use the deep-copied policies passed in
    # TODO: deep copy the environments so that it uses the right morphologies and paramters etc.
    _, thread_env = get_agents_and_env(debug=False)
    
    # Move policies to the appropriate device for this worker
    for p in worker_policies:
        p = p.to('cpu')
        p.eval()  # Ensure eval mode
    
    # Perform rollout with the copied policies and process-local env
    result = rollout(
        policy=worker_policies,
        env=thread_env,
        seeds=seeds,
        record_video=False,
        debug=False
    )
    
    # Clean up process-local environment
    thread_env.close()
    
    return result

def parallel_rollout(policy, env, seeds, N, merge=False):
    # Set multiprocessing start method to 'fork' since we currently do not plan on using CUDA ('spawn' better for CUDA)
    try:
        multiprocessing.set_start_method('fork', force=False)
    except RuntimeError:
        # Already set, ignore
        pass
    
    # Deep copy policies for each worker
    # Move to CPU first to ensure proper pickling across processes
    worker_policies_list = []
    for i in range(N):
        worker_policies = []
        for p in policy:
            p_copy = copy.deepcopy(p)
            p_copy = p_copy.cpu()
            worker_policies.append(p_copy)
        worker_policies_list.append(worker_policies)
    
    # Use multiprocessing instead of threading for true parallelism
    with multiprocessing.Pool(processes=N) as pool:
        # Each process processes all seeds with its own policy copy
        args_list = [
            (seeds, worker_policies)
            for worker_policies in worker_policies_list
        ]
        results = pool.map(_worker_rollout, args_list)
    
    if merge:
        return merge_parallel_rollouts(results)
    return results

def merge_parallel_rollouts(parallel_results):
    if not parallel_results:
        return []
    
    # Determine number of policies from first result
    num_policies = len(parallel_results[0])
    
    # Initialize merged result: one list per policy
    merged = [[] for _ in range(num_policies)]
    
    # For each policy, concatenate all episodes from all processes
    for policy_idx in range(num_policies):
        for process_result in parallel_results:
            # Extend this policy's list with all episodes from this process
            merged[policy_idx].extend(process_result[policy_idx])
    return merged

def count_steps_from_rollouts(rollouts):
    total_steps = 0
    
    if not rollouts:
        return 0
    
    # Count steps for just one policy (the first)
    if rollouts and len(rollouts) > 0:
        for episode in rollouts[0]:
            total_steps += len(episode.action) if episode.action else 0

    return total_steps






if __name__ == "__main__":
    import time
    
    # Set multiprocessing start method (required for some platforms)
    # 'spawn' is safer but slower, 'fork' is faster on Unix (Linux/Mac)
    multiprocessing.set_start_method('fork', force=True)

    policy, env = get_agents_and_env(debug=True)
    seeds = list(range(15))

    print("\n\n\n")
    print("Comparing single vs. parallel rollout")

    # Time single rollout
    start_time = time.time()
    single_result = single_rollout(policy, env, seeds)
    elapsed_single = time.time() - start_time
    single_steps = count_steps_from_rollouts(single_result)
    single_steps_per_sec = single_steps / elapsed_single if elapsed_single > 0 else 0

    # Time parallel rollouts
    n_values = [1, 2, 10, 16, 32]
    parallel_timings = []

    for N in tqdm(n_values, desc="Parallel rollout", leave=False):
        start_time = time.time()
        # parallel_results = parallel_rollout(policy, env, seeds, N)
        merged_results = parallel_rollout(policy, env, seeds, N, merge=True)
        elapsed = time.time() - start_time
        # Merge parallel results into single format for step counting
        # merged_results = merge_parallel_rollouts(parallel_results)
        parallel_steps = count_steps_from_rollouts(merged_results)
        parallel_steps_per_sec = parallel_steps / elapsed if elapsed > 0 else 0
        parallel_timings.append((N, elapsed, parallel_steps, parallel_steps_per_sec))

    # Print as table with aligned columns
    print(f"{'Scenario':<35} {'N':>3} {'Exec Time (s)':>16} {'Steps/sec':>12}")
    print(f"{'-'*35} {'-'*3} {'-'*16} {'-'*12}")
    print(f"{'Single rollout':<35} {'-':>3} {elapsed_single:16.2f} {single_steps_per_sec:12.0f}")
    for N, exec_time, steps, steps_per_sec in parallel_timings:
        print(f"{'Parallel rollout':<35} {N:>3} {exec_time:16.2f} {steps_per_sec:12.0f}")

