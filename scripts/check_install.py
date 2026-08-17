"""Print versions and confirm that MineRL registered the two target tasks."""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys


def command_version(command, version_argument="--version"):
    executable = shutil.which(command)
    if executable is None:
        return "not found"
    result = subprocess.run(
        [executable, version_argument], capture_output=True, text=True, check=False
    )
    return (result.stdout or result.stderr).strip()


def main():
    print("OS:", platform.platform())
    print("Python:", sys.version.replace("\n", " "))
    print("pip:", command_version("pip"))
    print("conda:", command_version("conda"))
    # Java 8 supports ``-version`` but not the newer ``--version`` spelling.
    print("Java:", command_version("java", "-version"))
    print("JAVA_HOME:", os.environ.get("JAVA_HOME", "not set"))
    print("Git:", command_version("git"))
    print("bash:", shutil.which("bash") or "not found")

    try:
        import gym
        import minerl  # noqa: F401

        print("gym:", importlib.metadata.version("gym"))
        print("MineRL:", importlib.metadata.version("minerl"))
        for env_id in ("MineRLNavigateDense-v0", "MineRLTreechop-v0"):
            gym.spec(env_id)
            print("registered:", env_id)
    except Exception as error:
        print("MineRL check failed: {}: {}".format(type(error).__name__, error))
        raise


if __name__ == "__main__":
    # Required by MineRL multiprocessing on Windows.
    main()
