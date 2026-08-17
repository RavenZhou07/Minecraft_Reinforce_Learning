"""Smoke-test reset -> observation -> action -> reward -> step -> close."""

import argparse

from mc_rl.envs import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="MineRLNavigateDense-v0")
    parser.add_argument("--max-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(
        args.env,
        discrete_actions=True,
        max_episode_steps=args.max_steps,
        log_path="logs/random_agent.csv",
    )
    print("observation space:", env.observation_space)
    print("raw MineRL action space:", env.unwrapped.action_space)
    print("wrapped action space:", env.action_space)

    cumulative_reward = 0.0
    try:
        observation = env.reset()
        print("reset succeeded; observation keys:", sorted(observation.keys()))
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            step += 1
            cumulative_reward += float(reward)
            if step == 1 or step % 10 == 0 or done:
                print(
                    "step={} action={} reward={:.3f} total={:.3f} done={}".format(
                        step, action, reward, cumulative_reward, done
                    )
                )
        print("episode ended normally; info:", info)
    finally:
        env.close()
        print("environment closed")


if __name__ == "__main__":
    main()
