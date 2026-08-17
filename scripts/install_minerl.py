"""Install MineRL 0.4.4 with one minimal historical-repository repair.

The 0.4.4 source asks JitPack for a MixinGradle commit that is no longer
resolvable. The source itself labels that commit as version 0.6. This script
downloads the unmodified PyPI sdist, replaces only that build dependency with
the surviving official SpongePowered 0.6-SNAPSHOT artifact, then asks pip to
build and install MineRL normally.
"""

import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import urlparse


MINERL_VERSION = "0.4.4"
BUILD_FILE = Path("minerl") / "Malmo" / "Minecraft" / "build.gradle"
OLD_REPOSITORY = "        maven { url 'https://jitpack.io' }\n"
NEW_REPOSITORY = OLD_REPOSITORY + (
    "        maven { url "
    "'https://repo.spongepowered.org/repository/maven-public/' }\n"
)
OLD_DEPENDENCY = "com.github.SpongePowered:MixinGradle:dcfaf61"
NEW_DEPENDENCY = "org.spongepowered:mixingradle:0.6-SNAPSHOT"
OLD_DECODE = "self.minecraft_process.stdout.readline().decode(mine_log_encoding)"
NEW_DECODE = (
    "self.minecraft_process.stdout.readline().decode("
    "mine_log_encoding, errors=\"replace\")"
)


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command))
    subprocess.run(command, check=True, **kwargs)


def require_compatible_runtime():
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("MineRL 0.4.4 setup requires this project's Python 3.8 env")
    java = shutil.which("java")
    if java is None:
        raise RuntimeError("java was not found; activate the Conda environment first")
    result = subprocess.run([java, "-version"], capture_output=True, text=True)
    version_text = result.stderr + result.stdout
    if 'version "1.8.' not in version_text:
        raise RuntimeError("MineRL 0.4.4 requires JDK 8; found:\n" + version_text)


def add_gradle_proxy_from_environment(env):
    """Java 8/Gradle does not automatically consume HTTPS_PROXY on Windows."""

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        return
    parsed = urlparse(proxy)
    if not parsed.hostname or not parsed.port:
        return
    options = (
        "-Dhttps.proxyHost={0} -Dhttps.proxyPort={1} "
        "-Dhttp.proxyHost={0} -Dhttp.proxyPort={1}"
    ).format(parsed.hostname, parsed.port)
    env["GRADLE_OPTS"] = (env.get("GRADLE_OPTS", "") + " " + options).strip()


def patch_build_file(source_root):
    build_file = source_root / BUILD_FILE
    original = build_file.read_text(encoding="utf-8")
    if OLD_DEPENDENCY not in original or OLD_REPOSITORY not in original:
        raise RuntimeError("unexpected MineRL build.gradle; refusing an ambiguous patch")
    patched = original.replace(OLD_REPOSITORY, NEW_REPOSITORY, 1)
    patched = patched.replace(OLD_DEPENDENCY, NEW_DEPENDENCY, 1)
    build_file.write_text(patched, encoding="utf-8")
    print("Patched MixinGradle artifact source in", build_file)


def patch_windows_log_decoding(source_root):
    """Prevent localized Java 8 logs from crashing Python UTF-8 mode."""

    malmo_file = source_root / "minerl" / "env" / "malmo.py"
    original = malmo_file.read_text(encoding="utf-8")
    if original.count(OLD_DECODE) != 1:
        raise RuntimeError("unexpected MineRL malmo.py; refusing an ambiguous patch")
    patched = original.replace(OLD_DECODE, NEW_DECODE, 1)
    # The background logger has two equivalent decode calls. They already
    # catch UnicodeDecodeError, but retrying with the same encoding can fail.
    patched = patched.replace(
        "line.decode(mine_log_encoding)",
        "line.decode(mine_log_encoding, errors=\"replace\")",
    )
    malmo_file.write_text(patched, encoding="utf-8")
    print("Patched localized Java log decoding in", malmo_file)


def main():
    require_compatible_runtime()
    try:
        if importlib.metadata.version("minerl") == MINERL_VERSION:
            print("MineRL {} is already installed.".format(MINERL_VERSION))
            return
    except importlib.metadata.PackageNotFoundError:
        pass

    child_env = os.environ.copy()
    add_gradle_proxy_from_environment(child_env)
    with tempfile.TemporaryDirectory(prefix="minerl-0.4.4-build-") as temp:
        temp_path = Path(temp)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "minerl==" + MINERL_VERSION,
                "--no-deps",
                "--dest",
                str(temp_path),
            ],
            env=child_env,
        )
        archive = next(temp_path.glob("minerl-0.4.4.tar.gz"))
        with tarfile.open(str(archive), "r:gz") as handle:
            handle.extractall(str(temp_path))
        source_root = temp_path / "minerl-0.4.4"
        patch_build_file(source_root)
        patch_windows_log_decoding(source_root)
        run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "--no-deps",
                str(source_root),
            ],
            env=child_env,
        )
    print("Installed MineRL", importlib.metadata.version("minerl"))


if __name__ == "__main__":
    main()
