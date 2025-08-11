# tests/test_collisions.py
import numpy.testing as npt
import pytest

from env import TwoPlayerPushEnv


@pytest.fixture
def env():
    e = TwoPlayerPushEnv()
    e.seed(0)  # make internal randomness deterministic for test runs
    e.reset()
    return e


def test_direct_horizontal_collision(env):
    """Tests direct horizontal collision between players and balls.

    All objects in same row:
    P1 at (4,2), B1 at (4,3), B2 at (4,4), P2 at (4,5)
    P1 moves right (4), P2 moves left (3) -> collision => nothing changes
    """
    env.p1_pos = (4, 2)
    env.p2_pos = (4, 5)
    env.b1_pos = (4, 3)
    env.b2_pos = (4, 4)

    obs, rewards, terminated, truncated, info = env.step([4, 3])

    npt.assert_array_equal(env.p1_pos, (4, 2), "P1 should remain in place")
    npt.assert_array_equal(env.p2_pos, (4, 5), "P2 should remain in place")
    npt.assert_array_equal(env.b1_pos, (4, 3), "Ball 1 should remain in place")
    npt.assert_array_equal(env.b2_pos, (4, 4), "Ball 2 should remain in place")
    assert max(rewards) == 0, "No points should be awarded"
    assert not terminated and not truncated, "Game should not end"


def test_vertical_collision(env):
    """Tests vertical collision between players and balls.

    Column-aligned scenario.
    P1 above B1, P2 below B2 in same column.
    P1 moves down (2), P2 moves up (1) -> collision => nothing changes
    """
    env.p1_pos = (2, 4)
    env.p2_pos = (5, 4)
    env.b1_pos = (3, 4)
    env.b2_pos = (4, 4)

    obs, rewards, terminated, truncated, info = env.step([2, 1])

    npt.assert_array_equal(env.p1_pos, (2, 4), "P1 should remain in place")
    npt.assert_array_equal(env.p2_pos, (5, 4), "P2 should remain in place")
    npt.assert_array_equal(env.b1_pos, (3, 4), "Ball 1 should remain in place")
    npt.assert_array_equal(env.b2_pos, (4, 4), "Ball 2 should remain in place")
    assert max(rewards) == 0, "No points should be awarded"
    assert not terminated and not truncated, "Game should not end"


def test_direct_horizontal_collision_with_gap():
    env = TwoPlayerPushEnv()

    possible_b1_positions = {tuple((3, 4)), tuple((5, 4))}
    possible_b2_positions = {tuple((3, 4)), tuple((5, 4))}

    for _ in range(10):
        env.reset()
        env.p1_pos = (4, 2)
        env.p2_pos = (4, 6)
        env.b1_pos = (4, 3)
        env.b2_pos = (4, 5)

        obs, rewards, terminated, truncated, info = env.step([4, 3])

        # Only check they're within allowed spots
        assert tuple(env.b1_pos) in possible_b1_positions, f"B1 invalid position: {env.b1_pos}"
        assert tuple(env.b2_pos) in possible_b2_positions, f"B2 invalid position: {env.b2_pos}"

        # Check fixed things
        npt.assert_array_equal(env.p1_pos, (4, 3))
        npt.assert_array_equal(env.p2_pos, (4, 5))
        assert max(rewards) == 0
        assert not terminated and not truncated


def test_direct_vertical_collision_with_gap():
    env = TwoPlayerPushEnv()

    possible_b1_positions = {tuple((4, 3)), tuple((4, 5))}
    possible_b2_positions = {tuple((4, 3)), tuple((4, 5))}

    for _ in range(10):
        env.reset()
        env.p1_pos = (2, 4)
        env.p2_pos = (6, 4)
        env.b1_pos = (3, 4)
        env.b2_pos = (5, 4)

        obs, rewards, terminated, truncated, info = env.step([2, 1])

        # Only check they're within allowed spots
        assert tuple(env.b1_pos) in possible_b1_positions, f"B1 invalid position: {env.b1_pos}"
        assert tuple(env.b2_pos) in possible_b2_positions, f"B2 invalid position: {env.b2_pos}"

        # Check fixed things
        npt.assert_array_equal(env.p1_pos, (3, 4))
        npt.assert_array_equal(env.p2_pos, (5, 4))
        assert max(rewards) == 0
        assert not terminated and not truncated
