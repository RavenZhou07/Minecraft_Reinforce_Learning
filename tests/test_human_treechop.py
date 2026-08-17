from argparse import Namespace

import numpy as np
import psutil
import pytest

from scripts.human_treechop import (
    action_for_key,
    close_env,
    draw_control_window,
    repeat_for_action,
    validate_args,
)


def test_human_control_key_mapping():
    assert action_for_key(ord("w")) == 1
    assert action_for_key(ord("W")) == 1
    assert action_for_key(ord(" ")) == 2
    assert action_for_key(ord("f")) == 7
    assert action_for_key(ord("q")) is None
    assert action_for_key(-1) is None


def test_human_control_action_repeats():
    assert repeat_for_action(1, move_repeat=5, attack_repeat=80) == 5
    assert repeat_for_action(7, move_repeat=5, attack_repeat=80) == 80
    assert repeat_for_action(8, move_repeat=5, attack_repeat=80) == 5
    assert repeat_for_action(3, move_repeat=5, attack_repeat=80) == 1


def test_human_control_preview_shape_and_dtype():
    pov = np.zeros((64, 64, 3), dtype=np.uint8)
    preview = draw_control_window(pov, 2, 100, 0.0, 7, 20)
    assert preview.shape == (662, 512, 3)
    assert preview.dtype == np.uint8


def test_human_control_rejects_invalid_arguments():
    args = Namespace(
        max_steps=0,
        move_repeat=5,
        attack_repeat=80,
        idle_delay_ms=100,
        active_delay_ms=50,
        save_every=25,
    )
    with pytest.raises(ValueError):
        validate_args(args)


def test_human_control_tolerates_only_already_exited_process(capsys):
    class AlreadyExitedEnv:
        @staticmethod
        def close():
            raise psutil.NoSuchProcess(123)

    close_env(AlreadyExitedEnv())
    assert "already exited" in capsys.readouterr().out


def test_human_control_does_not_hide_other_close_errors():
    class BrokenEnv:
        @staticmethod
        def close():
            raise RuntimeError("unexpected close failure")

    with pytest.raises(RuntimeError, match="unexpected close failure"):
        close_env(BrokenEnv())
