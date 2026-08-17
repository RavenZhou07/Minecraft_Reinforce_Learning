"""Exercise the Navigate observation/action/reward pipeline without training."""

import argparse
from pathlib import Path

from PIL import Image

from mc_rl.envs import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(
        "MineRLNavigateDense-v0",
        max_episode_steps=args.max_steps,
        log_path="logs/navigate_episodes.csv",
    )
    frame_dir = Path("logs/navigate_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Observation: a dict containing the 64x64 RGB POV, compass angle and dirt
    # inventory. Action: one of nine integers translated by our wrapper.
    # Reward: dense progress toward the compass target plus +100 at the goal.
    # Termination: goal reached, MineRL's time limit, or this demo's short limit.
    cumulative_reward = 0.0
    try:
        observation = env.reset()
        done = False
        step = 0
        while not done:
            # A deterministic pattern is easier to debug than a learned policy.
            action = 2 if step % 20 else 4  # forward+jump, periodically turn right
            observation, reward, done, info = env.step(action)
            step += 1
            cumulative_reward += float(reward)
            pov = observation["pov"]
            compass = observation["compass"]["angle"]
            print(
                "step={} pov={} compass={} reward={:.3f} total={:.3f} done={}".format(
                    step, pov.shape, compass, reward, cumulative_reward, done
                )
            )
            if args.save_every > 0 and step % args.save_every == 0:
                Image.fromarray(pov).save(frame_dir / "frame_{:05d}.png".format(step))
        print("episode info:", info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
