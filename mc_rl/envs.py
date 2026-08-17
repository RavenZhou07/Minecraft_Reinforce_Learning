"""Environment construction kept in one place for demos and future agents."""

import os
import sys
from typing import Optional
from urllib.parse import urlparse

import gym

from mc_rl.wrappers import (
    DiscreteActionWrapper,
    EpisodeCSVLogger,
    OneLogTreechopWrapper,
)


SUPPORTED_ENVIRONMENTS = ("MineRLNavigateDense-v0", "MineRLTreechop-v0")


def configure_minerl_runtime() -> None:
    """Keep old Gradle state local to each copied MineRL instance.

    MineRL launches Minecraft from a temporary copy whose ``run/gradle``
    directory was populated during installation. A relative value is resolved
    from that copied Minecraft directory by the launcher. This also avoids
    stale locks in the user's global ``.gradle`` directory.
    """

    os.environ.setdefault("GRADLE_USER_HOME", os.path.join("run", "gradle"))
    # Directly invoking the environment's python.exe does not run Conda's
    # activation scripts. Select the JDK 8 bundled inside this isolated env
    # for this process only, without touching global Java configuration.
    local_java_home = os.path.join(sys.prefix, "Library")
    local_java = os.path.join(local_java_home, "bin", "java.exe")
    if os.path.isfile(local_java):
        os.environ["JAVA_HOME"] = local_java_home
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        local_java_bin = os.path.dirname(local_java)
        if local_java_bin.lower() not in {
            entry.lower() for entry in path_entries if entry
        }:
            os.environ["PATH"] = local_java_bin + os.pathsep + os.environ.get("PATH", "")
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        parsed = urlparse(proxy)
        if parsed.hostname and parsed.port:
            marker = "-Dhttps.proxyHost={}".format(parsed.hostname)
            if marker not in os.environ.get("GRADLE_OPTS", ""):
                options = (
                    "-Dhttps.proxyHost={0} -Dhttps.proxyPort={1} "
                    "-Dhttp.proxyHost={0} -Dhttp.proxyPort={1}"
                ).format(parsed.hostname, parsed.port)
                os.environ["GRADLE_OPTS"] = (
                    os.environ.get("GRADLE_OPTS", "") + " " + options
                ).strip()


def make_env(
    env_id: str,
    discrete_actions: bool = True,
    max_episode_steps: Optional[int] = None,
    one_log_treechop: bool = False,
    log_path: Optional[str] = None,
) -> gym.Env:
    """Create a supported MineRL env and apply only the requested wrappers."""

    configure_minerl_runtime()
    # MineRL import registers all environment IDs and is expensive on this old
    # stack. Keep it local so unit tests and action utilities remain lightweight.
    import minerl  # noqa: F401

    if env_id not in SUPPORTED_ENVIRONMENTS:
        raise ValueError("unsupported environment: {}".format(env_id))
    if one_log_treechop and env_id != "MineRLTreechop-v0":
        raise ValueError("one_log_treechop is only valid for MineRLTreechop-v0")

    env = gym.make(env_id)
    if discrete_actions:
        env = DiscreteActionWrapper(env)

    if one_log_treechop:
        env = OneLogTreechopWrapper(env, max_episode_steps or 1000)
    elif max_episode_steps is not None:
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        # The MineRL specs already have long TimeLimits (6000/8000 steps).
        # This outer limit makes smoke tests deterministic and quick.
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if log_path is not None:
        env = EpisodeCSVLogger(env, log_path)
    return env
