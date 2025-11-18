#!/usr/bin/env python3
"""Speed test for video recording performance."""
import time
import argparse
from robosumo.envs.rollout import get_agents_and_env, rollout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-fast-mode', action='store_true', default=False,
                        help='Use fast video mode (lower quality, faster)')
    args = parser.parse_args()

    mode_str = "fast mode" if args.video_fast_mode else "normal mode"
    print(f"Running rollout with video recording ({mode_str})...")
    policies, env = get_agents_and_env(debug=False)
    
    start_time = time.time()
    rollouts = rollout(
        policy=policies,
        env=env,
        seeds=[42],
        record_video=True,
        debug=True,
        video_fast_mode=args.video_fast_mode,
    )
    elapsed = time.time() - start_time
    
    episode_length = len(rollouts[0][0].action) if rollouts and rollouts[0] else 0
    print(f"\nEpisode render time: {elapsed:.2f}s")
    print(f"Episode length: {episode_length} steps")
    print(f"Time per step: {elapsed/episode_length*1000:.2f}ms" if episode_length > 0 else "")

if __name__ == "__main__":
    main()

