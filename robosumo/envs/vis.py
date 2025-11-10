import io
import os

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

    for label, history in zip(agent_labels, value_histories):
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
    for label, history in zip(agent_labels, value_histories):
        steps = range(len(history))
        plt.plot(steps, history, label=label)
        if history:
            plt.scatter(len(history) - 1, history[-1], s=10)

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
):
    has_frames = bool(frames)
    has_values = bool(value_histories and value_histories[0])

    if not has_frames and not has_values:
        return

    os.makedirs(out_dir, exist_ok=True)

    if has_frames:
        video_path = os.path.join(out_dir, f"robosumo_episode{episode_idx}.mp4")
        if debug:
            print(f"Saving video to {video_path}...")
        imageio.mimsave(video_path, frames, fps=fps)
        if debug:
            print("Video saved successfully!")

    if not has_values:
        return

    total_steps = len(value_histories[0])
    y_limits = compute_value_axis_limits(value_histories)

    composite_frames = []
    if has_frames:
        for step_idx, frame_np in enumerate(frames):
            partial_histories = [
                hist[: step_idx + 1] for hist in value_histories
            ]
            value_plot = render_value_plot(
                agent_labels,
                partial_histories,
                current_step=step_idx,
                total_timesteps=total_steps,
                y_limits=y_limits,
            )
            if value_plot is None:
                continue

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

    if composite_frames:
        composite_path = os.path.join(out_dir, f"robosumo_episode{episode_idx}_values_video.mp4")
        if debug:
            print(f"Saving composite video to {composite_path}...")
        imageio.mimsave(composite_path, composite_frames, fps=fps)
        if debug:
            print("Composite video saved successfully!")

    if save_plot:
        plot_path = os.path.join(out_dir, f"robosumo_episode{episode_idx}_values.png")
        plot_values(agent_labels, value_histories, plot_path)