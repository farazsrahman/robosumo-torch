
import gymnasium as gym
import os
import time
import torch
from dataclasses import dataclass, field
from typing import List
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
    seeds, worker_policies, record_video, video_dir, debug = args
    
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
        record_video=record_video,
        video_fast_mode=True,
        debug=debug,
        video_dir=video_dir
    )
    
    # Clean up process-local environment
    thread_env.close()
    
    return result

def parallel_rollout(
    policy, 
    env, 
    seeds: List[List[int]], 
    merge=True, 
    record_video=False, 
    debug=False, 
    video_fast_mode=False, 
    video_dir="out"
):
    # The number of processes corresponds to the number of seed lists
    N = len(seeds)
    try:
        multiprocessing.set_start_method('fork', force=False)
    except RuntimeError:
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
    
    # Each process gets its own list of seeds (seeds[i])
    with multiprocessing.Pool(processes=N) as pool:
        args_list = [
            (
                seeds[i], 
                worker_policies_list[i], 
                record_video and i == 0, # HACK (faraz): only record on one of the thread so the recordings do NOT all override eachother
                video_dir, 
                debug
            ) for i in range(N)
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
    seeds = list(range(3))

    print("\n\n\n")
    print("Comparing single vs. parallel rollout")

    # Helper to compute average steps per episode from rollouts (for first policy)
    def average_steps_per_episode(rollouts):
        if not rollouts or len(rollouts[0]) == 0:
            return 0
        lengths = [len(ep.action) for ep in rollouts[0] if hasattr(ep, "action") and ep.action is not None]
        if not lengths:
            return 0
        return sum(lengths) / len(lengths)

    # Time single rollout
    start_time = time.time()
    single_result = single_rollout(policy, env, seeds)
    elapsed_single = time.time() - start_time
    single_steps = count_steps_from_rollouts(single_result)
    single_steps_per_sec = single_steps / elapsed_single if elapsed_single > 0 else 0
    single_avg_steps = average_steps_per_episode(single_result)

    # Time parallel rollouts
    n_values = [1, 2, 10, 16, 32]
    parallel_timings = []

    for N in tqdm(n_values, desc="Parallel rollout", leave=False):
        start_time = time.time()

        merged_results = parallel_rollout(policy, env, [seeds]*N, record_video=True, merge=True, debug=False)
        
        elapsed = time.time() - start_time
        parallel_steps = count_steps_from_rollouts(merged_results)
        parallel_steps_per_sec = parallel_steps / elapsed if elapsed > 0 else 0
        parallel_avg_steps = average_steps_per_episode(merged_results)
        parallel_timings.append((N, elapsed, parallel_steps, parallel_steps_per_sec, parallel_avg_steps))

    # Print as table with aligned columns, now including avg steps/episode
    print(f"{'Scenario':<35} {'N':>3} {'Exec Time (s)':>16} {'Steps/sec':>12} {'Avg steps/ep':>14}")
    print(f"{'-'*35} {'-'*3} {'-'*16} {'-'*12} {'-'*14}")
    print(f"{'Single rollout':<35} {'-':>3} {elapsed_single:16.2f} {single_steps_per_sec:12.0f} {single_avg_steps:14.2f}")
    for N, exec_time, steps, steps_per_sec, avg_steps in parallel_timings:
        print(f"{'Parallel rollout':<35} {N:>3} {exec_time:16.2f} {steps_per_sec:12.0f} {avg_steps:14.2f}")

