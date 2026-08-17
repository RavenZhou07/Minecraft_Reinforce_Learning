"""Exercise movement, attacks, camera and rewards in MineRLTreechop."""

import argparse
from pathlib import Path

from PIL import Image

from mc_rl.actions import ACTION_NAMES
from mc_rl.envs import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(
        "MineRLTreechop-v0",
        max_episode_steps=args.max_steps,
        one_log_treechop=True,
        log_path="logs/treechop_episodes.csv",
    )
    frame_dir = Path("logs/treechop_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)

    # The raw observation stays intact; Treechop's learning input can therefore
    # be observation["pov"]. Reward is +1 per acquired log. OneLogTreechop ends
    # immediately on the first positive reward, or at --max-steps.
    pattern = [1, 8, 8, 7, 3, 5, 4, 6, 2]
    cumulative_reward = 0.0
    try:
        observation = env.reset()
        done = False
        step = 0
        while not done:
            action = pattern[step % len(pattern)]
            observation, reward, done, info = env.step(action)
            step += 1
            cumulative_reward += float(reward)
            pov = observation["pov"]
            print(
                "step={} pov={} action={} reward={:.3f} total={:.3f} done={} success={}".format(
                    step,
                    pov.shape,
                    ACTION_NAMES[action],
                    reward,
                    cumulative_reward,
                    done,
                    info.get("success", False),
                )
            )
            if args.save_every > 0 and step % args.save_every == 0:
                Image.fromarray(pov).save(frame_dir / "frame_{:05d}.png".format(step))
        print("episode info:", info)
    finally:
        env.close()


if __name__ == "__main__":
    main()
