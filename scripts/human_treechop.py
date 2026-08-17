"""Control MineRLTreechop from a live POV window.

This is a positive-reward sanity check, not an RL agent. The human observes
the same ``obs["pov"]`` pixels that a future policy will receive and chooses
only actions from the project's Discrete(9) action space.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image

from mc_rl.actions import ACTION_NAMES
from mc_rl.envs import make_env


KEY_TO_ACTION = {
    ord("n"): 0,  # noop / cancel the current repeated action
    ord("w"): 1,  # walk forward
    ord(" "): 2,  # walk forward and jump
    ord("a"): 3,  # turn left
    ord("d"): 4,  # turn right
    ord("i"): 5,  # look up
    ord("k"): 6,  # look down
    ord("f"): 7,  # attack; repeated long enough to chop by hand
    ord("g"): 8,  # walk forward while attacking
}
QUIT_KEYS = (27, ord("q"), ord("Q"))  # Escape or q
WINDOW_NAME = "MineRL human control"

TRANSITION_FIELDS = (
    "step",
    "action_id",
    "action_name",
    "reward",
    "cumulative_reward",
    "done",
    "success",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Human-control positive-reward check for MineRLTreechop."
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--move-repeat",
        type=int,
        default=5,
        help="Steps triggered by w, Space, or g (default: 5).",
    )
    parser.add_argument(
        "--attack-repeat",
        type=int,
        default=80,
        help="Steps triggered by f; another key can interrupt it (default: 80).",
    )
    parser.add_argument(
        "--idle-delay-ms",
        type=int,
        default=100,
        help="Time to wait for a key before sending an idle noop.",
    )
    parser.add_argument(
        "--active-delay-ms",
        type=int,
        default=50,
        help="Delay between repeated movement/attack steps.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save a POV frame every N environment steps; 0 disables it.",
    )
    parser.add_argument("--output-dir", default="logs/human_treechop")
    return parser.parse_args()


def action_for_key(key):
    """Map an OpenCV key code to an action ID, accepting upper-case letters."""

    key = key & 0xFF
    if ord("A") <= key <= ord("Z"):
        key += ord("a") - ord("A")
    return KEY_TO_ACTION.get(key)


def repeat_for_action(action_id, move_repeat, attack_repeat):
    """Return how many steps a single key press should remain active."""

    if action_id == 7:
        return attack_repeat
    if action_id in (1, 2, 8):
        return move_repeat
    return 1


def draw_control_window(pov, step, max_steps, reward, action_id, remaining):
    """Build a readable 512-pixel preview without changing the observation."""

    # MineRL returns RGB. OpenCV windows expect BGR. The conversion is only
    # for display; the original observation passed to logging stays untouched.
    bgr = cv2.cvtColor(pov, cv2.COLOR_RGB2BGR)
    preview = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_NEAREST)
    panel = np.zeros((150, 512, 3), dtype=np.uint8)
    lines = (
        "W forward | Space jump | A/D turn | I/K look",
        "F attack | G forward+attack | N cancel/noop | Q/Esc quit",
        "step {}/{}  reward {:.1f}".format(step, max_steps, reward),
        "action {}  repeats left {}".format(ACTION_NAMES[action_id], remaining),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            panel,
            line,
            (10, 28 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((preview, panel))


def validate_args(args):
    values = (
        args.max_steps,
        args.move_repeat,
        args.attack_repeat,
        args.idle_delay_ms,
        args.active_delay_ms,
    )
    if any(value <= 0 for value in values):
        raise ValueError("step, repeat, and delay arguments must be positive")
    if args.save_every < 0:
        raise ValueError("save-every cannot be negative")


def close_env(env):
    """Close MineRL while tolerating only its known already-exited PID race."""

    try:
        env.close()
    except psutil.NoSuchProcess as error:
        # A successful terminal mission can close Minecraft before MineRL's
        # Windows cleanup code reaps it. The target process is already gone,
        # so this one exception is safe to report and continue past. All other
        # close errors deliberately remain visible.
        print(
            "WARNING: Minecraft had already exited during MineRL close: {}".format(
                error
            ),
            flush=True,
        )


def main():
    args = parse_args()
    validate_args(args)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    session_dir = Path(args.output_dir) / session_id
    frame_dir = session_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    transitions_path = session_dir / "transitions.csv"
    summary_path = session_dir / "summary.json"

    env = make_env(
        "MineRLTreechop-v0",
        discrete_actions=True,
        max_episode_steps=args.max_steps,
        one_log_treechop=True,
    )
    # Human tests are diagnostic: surface a Minecraft failure instead of
    # silently launching a replacement instance during the same session.
    env.unwrapped._is_fault_tolerant = False
    env.seed(args.seed)

    print("Controls:")
    print("  W forward | Space forward+jump | A/D turn | I/K look")
    print("  F attack  | G forward+attack   | N cancel/noop | Q/Esc quit")
    print("Click the 'MineRL human control' POV window before pressing keys.")
    print("A new key interrupts the currently repeated action.")

    observation = None
    step = 0
    cumulative_reward = 0.0
    active_action = 0
    repeats_left = 0
    success = False
    termination = "not_started"
    started_at = time.perf_counter()

    try:
        observation = env.reset()
        Image.fromarray(observation["pov"]).save(frame_dir / "initial.png")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        with transitions_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRANSITION_FIELDS)
            writer.writeheader()

            done = False
            info = {}
            while not done:
                window = draw_control_window(
                    observation["pov"],
                    step,
                    args.max_steps,
                    cumulative_reward,
                    active_action,
                    repeats_left,
                )
                cv2.imshow(WINDOW_NAME, window)

                delay = (
                    args.active_delay_ms if repeats_left > 0 else args.idle_delay_ms
                )
                key = cv2.waitKey(delay)
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    termination = "human_quit"
                    break
                normalized_key = key & 0xFF
                if normalized_key in QUIT_KEYS:
                    termination = "human_quit"
                    break

                requested_action = action_for_key(key) if key != -1 else None
                if requested_action is not None:
                    active_action = requested_action
                    repeats_left = repeat_for_action(
                        active_action, args.move_repeat, args.attack_repeat
                    )
                elif repeats_left <= 0:
                    active_action = 0

                action = active_action
                observation, reward, done, info = env.step(action)
                step += 1
                cumulative_reward += float(reward)
                if repeats_left > 0:
                    repeats_left -= 1

                success = bool(info.get("success", False))
                writer.writerow(
                    {
                        "step": step,
                        "action_id": action,
                        "action_name": ACTION_NAMES[action],
                        "reward": float(reward),
                        "cumulative_reward": cumulative_reward,
                        "done": bool(done),
                        "success": success,
                    }
                )
                handle.flush()

                if args.save_every > 0 and step % args.save_every == 0:
                    Image.fromarray(observation["pov"]).save(
                        frame_dir / "frame_{:05d}.png".format(step)
                    )
                if reward != 0 or done:
                    print(
                        "step={} action={} reward={:.3f} total={:.3f} "
                        "done={} success={}".format(
                            step,
                            ACTION_NAMES[action],
                            reward,
                            cumulative_reward,
                            done,
                            success,
                        ),
                        flush=True,
                    )

            if success:
                termination = "one_log"
            elif done and info.get("TimeLimit.truncated", False):
                termination = "step_limit"
            elif done:
                termination = "minerl_done"
    finally:
        if observation is not None:
            Image.fromarray(observation["pov"]).save(frame_dir / "terminal.png")
        cv2.destroyAllWindows()
        close_env(env)

    summary = {
        "session_id": session_id,
        "seed": args.seed,
        "steps": step,
        "cumulative_reward": cumulative_reward,
        "success": success,
        "termination": termination,
        "duration_seconds": round(time.perf_counter() - started_at, 3),
        "transitions_csv": str(transitions_path),
        "frames_directory": str(frame_dir),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("session directory:", session_dir)


if __name__ == "__main__":
    main()
