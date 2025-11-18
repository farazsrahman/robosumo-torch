import io
import os
import time

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_agent_labels(policy):
    return [
        f"Agent {idx} ({getattr(policy[idx], 'morphology', 'unknown')})"
        for idx in range(len(policy))
    ]


def compute_value_axis_limits(value_histories):
    if not value_histories or not value_histories[0]:
        return None

    all_values = np.concatenate(
        [np.array(hist, dtype=np.float32) for hist in value_histories if hist],
        axis=0,
    ) if any(len(hist) > 0 for hist in value_histories) else np.array([], dtype=np.float32)

    if all_values.size == 0:
        return None

    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))

    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    return (y_min, y_max)


def plot_values(agent_labels, value_histories, plot_path):
    if not value_histories or not value_histories[0]:
        return

    timesteps = range(len(value_histories[0]))
    plt.figure()

    for idx, (label, history) in enumerate(zip(agent_labels, value_histories)):
        # colors to match those asigned to the bodies in MuJoco
        color = "red" if idx == 0 else "green" if idx == 1 else None
        if color:
            plt.plot(timesteps, history, label=label, color=color)
        else:
            plt.plot(timesteps, history, label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Value Prediction")
    plt.title("Value Predictions per Agent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def render_value_plot(agent_labels, value_histories, current_step, total_timesteps, y_limits=None):
    """Return a PIL Image visualizing value histories up to the current timestep."""
    if not value_histories or total_timesteps <= 0:
        return None

    plt.figure()
    for idx, (label, history) in enumerate(zip(agent_labels, value_histories)):
        steps = range(len(history))
        # colors to match those asigned to the bodies in MuJoco
        color = "red" if idx == 0 else "green" if idx == 1 else None
        if color:
            plt.plot(steps, history, label=label, color=color)
        else:
            plt.plot(steps, history, label=label)
        if history:
            scatter_kwargs = {"s": 10}
            if color:
                scatter_kwargs["color"] = color
            plt.scatter(len(history) - 1, history[-1], **scatter_kwargs)

    plt.xlabel("Timestep")
    plt.ylabel("Value Prediction")
    plt.title("Live Value Predictions")
    plt.legend()
    plt.xlim(0, max(total_timesteps - 1, 1))

    if y_limits is not None:
        plt.ylim(*y_limits)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")
    buffer.close()
    return image


def save_video_w_value(
    episode_idx,
    frames,
    value_histories,
    agent_labels,
    out_dir,
    fps=30,
    debug=False,
    save_plot=True,
    profile=False,
    animate_plot=False,  # If False, use static plot for entire trajectory (much faster)
):
    has_frames = bool(frames)
    has_values = bool(value_histories and value_histories[0])

    if not has_frames and not has_values:
        return

    os.makedirs(out_dir, exist_ok=True)

    if has_frames:
        if profile:
            print(f"\n=== Video Profile for Episode {episode_idx} ===")
            print(f"  Input frames: {len(frames)} frames, size: {frames[0].shape[1]}x{frames[0].shape[0]}")
        
        video_path = os.path.join(out_dir, f"robosumo_episode{episode_idx}.mp4")
        if debug:
            print(f"Saving video to {video_path}...")
        
        start_save = time.time()
        imageio.mimsave(video_path, frames, fps=fps)
        save_time = time.time() - start_save
        
        if profile:
            print(f"  Video encoding: {save_time:.3f}s ({save_time/len(frames)*1000:.2f}ms per frame)")
            print(f"  Total video save: {save_time:.3f}s")
            print(f"  Output size: {frames[0].shape[1]}x{frames[0].shape[0]}")
        
        if debug:
            print("Video saved successfully!")

    if not has_values:
        return

    total_steps = len(value_histories[0])
    y_limits = compute_value_axis_limits(value_histories)

    composite_frames = []
    if has_frames:
        if profile:
            print(f"\n=== Composite Video Profile for Episode {episode_idx} ===")
            mode_str = "animated" if animate_plot else "static"
            print(f"  Plot mode: {mode_str}")
        
        start_composite = time.time()
        plot_render_time = 0
        composite_build_time = 0
        
        # Render plot once if using static mode (much faster)
        static_value_plot = None
        if not animate_plot:
            plot_start = time.time()
            # Use full trajectory for static plot
            static_value_plot = render_value_plot(
                agent_labels,
                value_histories,  # Full histories, not partial
                current_step=total_steps - 1,
                total_timesteps=total_steps,
                y_limits=y_limits,
            )
            plot_render_time = time.time() - plot_start
            if profile:
                print(f"  Static plot rendering: {plot_render_time:.3f}s (one-time)")
        
        for step_idx, frame_np in enumerate(frames):
            # Use static plot if available, otherwise render animated plot
            if animate_plot:
                partial_histories = [
                    hist[: step_idx + 1] for hist in value_histories
                ]
                
                plot_start = time.time()
                value_plot = render_value_plot(
                    agent_labels,
                    partial_histories,
                    current_step=step_idx,
                    total_timesteps=total_steps,
                    y_limits=y_limits,
                )
                plot_render_time += time.time() - plot_start
            else:
                value_plot = static_value_plot
            
            if value_plot is None:
                continue

            build_start = time.time()
            frame_array = np.array(frame_np)
            plot_height = value_plot.height
            frame_height = frame_array.shape[0]
            if plot_height != frame_height:
                new_width = int(value_plot.width * frame_height / plot_height)
                value_plot = value_plot.resize((new_width, frame_height), Image.BILINEAR)
            value_array = np.array(value_plot)

            min_height = min(frame_array.shape[0], value_array.shape[0])
            frame_array = frame_array[:min_height, :, ...]
            value_array = value_array[:min_height, :, ...]

            composite_frame = np.concatenate((frame_array, value_array), axis=1)
            composite_frames.append(composite_frame)
            composite_build_time += time.time() - build_start

        if profile and composite_frames:
            if animate_plot:
                print(f"  Plot rendering: {plot_render_time:.3f}s ({plot_render_time/len(composite_frames)*1000:.2f}ms per frame)")
            else:
                print(f"  Plot rendering: {plot_render_time:.3f}s (static, one-time)")
            print(f"  Composite frame building: {composite_build_time:.3f}s ({composite_build_time/len(composite_frames)*1000:.2f}ms per frame)")

    if composite_frames:
        composite_path = os.path.join(out_dir, f"robosumo_episode{episode_idx}_values_video.mp4")
        if debug:
            print(f"Saving composite video to {composite_path}...")
        
        start_composite_save = time.time()
        imageio.mimsave(composite_path, composite_frames, fps=fps)
        composite_save_time = time.time() - start_composite_save
        
        if profile:
            total_composite_time = time.time() - start_composite
            print(f"  Composite video encoding: {composite_save_time:.3f}s ({composite_save_time/len(composite_frames)*1000:.2f}ms per frame)")
            print(f"  Total composite video: {total_composite_time:.3f}s")
        
        if debug:
            print("Composite video saved successfully!")

    if save_plot:
        plot_path = os.path.join(out_dir, f"robosumo_episode{episode_idx}_values.png")
        plot_values(agent_labels, value_histories, plot_path)