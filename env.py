"""
Two-player competitive pushing game environment implementation.

This module implements a custom Gymnasium environment where two players compete
to push balls off opposite edges of a grid. The environment follows the Gymnasium
interface and provides both single and multi-agent compatible APIs.

Key Features:
    - 8x8 grid world (configurable size)
    - Two players with 5 actions each (stay, up, down, left, right)
    - Two balls that can be pushed
    - Scoring system based on pushing balls off edges
    - Complex collision resolution system
"""

import random
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Action mapping
# 0: stay, 1: up, 2: down, 3: left, 4: right
ACTION_DELTA = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def add_tuples(a: tuple, b: tuple) -> tuple:
    """Add two tuples element-wise.

    Args:
        a: First tuple
        b: Second tuple

    Returns:
        Tuple with element-wise sum
    """
    return (a[0] + b[0], a[1] + b[1])


def in_bounds(pos: tuple, size: int) -> bool:
    """Check if position is within grid bounds.

    Args:
        pos: Position tuple (row, col)
        size: Grid size

    Returns:
        True if position is within bounds
    """
    return 0 <= pos[0] < size and 0 <= pos[1] < size


class TwoPlayerPushEnv(gym.Env):
    """Two-player competitive pushing game environment.

    Players compete to push balls off opposite edges of an 8x8 grid.
    Scoring occurs when a ball falls off a player's target edge.

    Args:
        grid_size (int): Size of the square grid. Defaults to 8.
        max_steps (int): Maximum episode length. Defaults to 100.
        goal_score (int): Score needed to win. Defaults to 20.
        seed (Optional[int]): Random seed. Defaults to None.

    Attributes:
        observation_space (Box): 8x8x4 binary grid showing positions
        action_space (MultiDiscrete): [5,5] for both players' actions
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 100,
        goal_score: int = 20,
        seed: Optional[int] = None,
    ):
        """Initialize environment.

        Args:
            grid_size: Size of the square grid
            max_steps: Maximum episode length
            goal_score: Score needed to win
            seed: Random seed
        """
        super().__init__()
        self.size = grid_size
        self.max_steps = max_steps
        self.goal_score = goal_score

        # Observation: 4-channel binary grid
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.size, self.size, 4), dtype=np.int8)
        # Actions: both players choose an action among 5
        self.action_space = spaces.MultiDiscrete([5, 5])

        # Game state
        self.p1_pos = None
        self.p2_pos = None
        self.b1_pos = None
        self.b2_pos = None

        self.p1_score = 0
        self.p2_score = 0
        self.step_count = 0

        # RNG
        self.np_random = np.random.RandomState()
        self._seed_val = None
        if seed is not None:
            self.seed(seed)

        self.reset()

    def seed(self, seed: Optional[int] = None) -> list:
        """Set random seed.

        Args:
            seed: Random seed value

        Returns:
            List containing the seed
        """
        self._seed_val = seed if seed is not None else random.randrange(2**32)
        self.np_random.seed(self._seed_val)
        random.seed(self._seed_val)
        return [self._seed_val]

    def _empty_cells(self):
        occ = {self.p1_pos, self.p2_pos, self.b1_pos, self.b2_pos}
        empties = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in occ]
        return empties

    def _place_random_empty(self):
        empties = self._empty_cells()
        if not empties:
            # Should be rare; return a random cell (will collide) if no empties
            return (
                self.np_random.randint(self.size),
                self.np_random.randint(self.size),
            )
        return tuple(
            self.np_random.choice(len(empties)) and empties[0]
            if False
            else empties[self.np_random.randint(len(empties))]
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Initial observation and empty info dict
        """
        if seed is not None:
            self.seed(seed)

        # Random initial positions, ensuring uniqueness
        all_positions = [(r, c) for r in range(self.size) for c in range(self.size)]
        self.np_random.shuffle(all_positions)
        it = iter(all_positions)
        self.p1_pos = next(it)
        self.p2_pos = next(it)
        self.b1_pos = next(it)
        self.b2_pos = next(it)

        self.p1_score = 0
        self.p2_score = 0
        self.step_count = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        grid = np.zeros((self.size, self.size, 4), dtype=np.int8)
        grid[self.p1_pos[0], self.p1_pos[1], 0] = 1
        grid[self.p2_pos[0], self.p2_pos[1], 1] = 1
        grid[self.b1_pos[0], self.b1_pos[1], 2] = 1
        grid[self.b2_pos[0], self.b2_pos[1], 3] = 1
        return grid

    def _respawn_ball(self, ball_index: int):
        # Place ball randomly in an empty cell
        # old_positions = {self.p1_pos, self.p2_pos, self.b1_pos, self.b2_pos}

        # remove the ball's current pos so it can potentially reuse it?
        # Appears random spot given there are no objects there.
        occ = {self.p1_pos, self.p2_pos}
        if ball_index == 1:
            occ.add(self.b2_pos)
        else:
            occ.add(self.b1_pos)
        empties = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in occ]
        if not empties:
            # fallback: random position
            newpos = (
                self.np_random.randint(self.size),
                self.np_random.randint(self.size),
            )
        else:
            newpos = empties[self.np_random.randint(len(empties))]
        if ball_index == 1:
            self.b1_pos = newpos
        else:
            self.b2_pos = newpos

    def _apply_bounds_and_score(self, pos: Tuple[int, int], ball_index: int) -> Tuple[Optional[Tuple[int, int]], int]:
        """
        If pos is out of bounds, handle scoring & respawn. Return (new_pos or None if fell), score_gained.

        For scoring: p1 gets a point if their ball falls off LEFT edge (col < 0).
                     p2 gets a point if their ball falls off RIGHT edge (col >= size).
        For any ball falling off any edge, respawn at random empty cell.
        """
        r, c = pos
        score = 0

        if 0 <= r < self.size and 0 <= c < self.size:
            return (pos, 0)

        if ball_index == 1:
            # player1's goal: left edge (c < 0)
            if c < 0:
                score = 1
        else:
            # player2's goal: right edge (c >= size)
            if c >= self.size:
                score = 1
        # respawn ball
        self._respawn_ball(ball_index)
        return (None, score)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """Execute one environment step.

        Args:
            action: Array with actions for both players [p1_action, p2_action]

        Returns:
            observation: Current game state
            reward: [p1_reward, p2_reward]
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        if self.step_count >= self.max_steps:
            raise RuntimeError(
                f"Calling step() after episode end. Current steps: {self.step_count}, Max steps: {self.max_steps}"
            )

        a1, a2 = int(action[0]), int(action[1])
        self.step_count += 1

        # Current positions
        prev_p1 = self.p1_pos
        prev_p2 = self.p2_pos
        prev_b1 = self.b1_pos
        prev_b2 = self.b2_pos

        # Compute desired positions (players)
        dp1 = ACTION_DELTA[a1]
        dp2 = ACTION_DELTA[a2]
        desired_p1 = add_tuples(self.p1_pos, dp1)
        desired_p2 = add_tuples(self.p2_pos, dp2)

        # clamp desired positions to in-bounds for 'moving into cells' check
        # (we allow desired out-of-bounds as an attempt to move off board)
        # but players cannot leave board; if desired out of bounds, they stay instead
        if not in_bounds(desired_p1, self.size):
            desired_p1 = self.p1_pos
            dp1 = (0, 0)
        if not in_bounds(desired_p2, self.size):
            desired_p2 = self.p2_pos
            dp2 = (0, 0)

        # Helper flags
        p1_wants_b1 = desired_p1 == self.b1_pos
        p1_wants_b2 = desired_p1 == self.b2_pos
        p2_wants_b1 = desired_p2 == self.b1_pos
        p2_wants_b2 = desired_p2 == self.b2_pos

        # Step resolution plan:
        # 1) Resolve both players competing for same empty cell -> random 50/50,
        #   other stays.
        # 2) Resolve players pushing balls:
        #    - Determine which player pushes which ball.
        #    - If both push same ball: owner gets priority; otherwise random.
        # 3) Apply ball intended movements and resolve ball-ball collisions
        #   (perpendicular split or revert on swap).
        # 4) Apply players movement (some may be blocked/stay).
        #
        # We'll proceed carefully capturing intended moves for balls & players.

        # ---------- Resolve players wanting same cell (non-ball) ----------
        players_want_same_cell = (
            (desired_p1 == desired_p2) and (desired_p1 != self.b1_pos) and (desired_p1 != self.b2_pos)
        )
        p1_moves = True
        p2_moves = True
        if players_want_same_cell:
            # random 50-50
            chooser = self.np_random.randint(2)
            if chooser == 0:
                p2_moves = False
                desired_p2 = self.p2_pos
                dp2 = (0, 0)
            else:
                p1_moves = False
                desired_p1 = self.p1_pos
                dp1 = (0, 0)

        # ---------- Determine push intents ----------
        # For each ball, list who attempts to push it and from which direction
        # A push attempt happens when a player moves into
        # the ball's cell (desired == ball_pos)
        b1_pushers = []
        b2_pushers = []
        if p1_wants_b1 and p1_moves:
            b1_pushers.append(("p1", dp1))
        if p2_wants_b1 and p2_moves:
            b1_pushers.append(("p2", dp2))
        if p1_wants_b2 and p1_moves:
            b2_pushers.append(("p1", dp1))
        if p2_wants_b2 and p2_moves:
            b2_pushers.append(("p2", dp2))

        # Decide who pushes each ball
        # For ball 1 owner is player1, for ball 2 owner is player2
        b1_will_be_pushed_by = None
        b1_push_delta = (0, 0)
        if len(b1_pushers) == 1:
            b1_will_be_pushed_by = b1_pushers[0][0]
            b1_push_delta = b1_pushers[0][1]
        elif len(b1_pushers) == 2:
            # if one pusher is the owner (p1), owner priority
            if any(p[0] == "p1" for p in b1_pushers) and any(p[0] == "p2" for p in b1_pushers):
                # owner (p1) gets priority
                for p in b1_pushers:
                    if p[0] == "p1":
                        b1_will_be_pushed_by = "p1"
                        b1_push_delta = p[1]
                        break
            else:
                # neither is owner (shouldn't happen for b1 because owner is p1),
                # fallback random
                choose = self.np_random.randint(2)
                b1_will_be_pushed_by = b1_pushers[choose][0]
                b1_push_delta = b1_pushers[choose][1]

        b2_will_be_pushed_by = None
        b2_push_delta = (0, 0)
        if len(b2_pushers) == 1:
            b2_will_be_pushed_by = b2_pushers[0][0]
            b2_push_delta = b2_pushers[0][1]
        elif len(b2_pushers) == 2:
            if any(p[0] == "p2" for p in b2_pushers) and any(p[0] == "p1" for p in b2_pushers):
                # owner p2 priority
                for p in b2_pushers:
                    if p[0] == "p2":
                        b2_will_be_pushed_by = "p2"
                        b2_push_delta = p[1]
                        break
            else:
                choose = self.np_random.randint(2)
                b2_will_be_pushed_by = b2_pushers[choose][0]
                b2_push_delta = b2_pushers[choose][1]

        # If a ball has no pushers, but a player moves into its cell from a direction
        # that isn't same-cell? That can't happen.

        # Compute intended new ball positions (before resolving ball-ball collisions)
        intended_b1 = self.b1_pos
        intended_b2 = self.b2_pos
        b1_moves = False
        b2_moves = False
        if b1_will_be_pushed_by is not None:
            intended_b1 = add_tuples(self.b1_pos, b1_push_delta)
            # If intended_b1 equals previous b2_pos and intended_b2 equals previous b1_pos
            # -> potential swap; we'll handle later.
            b1_moves = True
        if b2_will_be_pushed_by is not None:
            intended_b2 = add_tuples(self.b2_pos, b2_push_delta)
            b2_moves = True

        # Prevent ball swap (skipping each other)
        # If balls attempt to move into each other's previous positions (swap)
        # and they were in same row or same column, then keep both in place.
        swap_attempt = (
            b1_moves
            and b2_moves
            and intended_b1 == prev_b2
            and intended_b2 == prev_b1
            and (prev_b1[0] == prev_b2[0] or prev_b1[1] == prev_b2[1])
        )
        if swap_attempt:
            # Cancel both ball moves
            intended_b1 = prev_b1
            intended_b2 = prev_b2
            b1_moves = b2_moves = False

        # Resolve ball-ball same-target collisions
        if b1_moves and b2_moves and intended_b1 == intended_b2:
            # For reproducibility, sample two coin flips
            flip_direction = self.np_random.randint(2)  # 0: try vertical first, 1: try horizontal first
            flip_balls = self.np_random.randint(2)  # 0: b1 left/up, b2 right/down, 1: vice versa

            # Try both vertical and horizontal splits
            cand_up = add_tuples(intended_b1, (-1, 0))
            cand_down = add_tuples(intended_b1, (1, 0))
            cand_left = add_tuples(intended_b1, (0, -1))
            cand_right = add_tuples(intended_b1, (0, 1))

            # Prepare attempts based on random ball assignment
            if flip_direction == 0:
                # Try vertical splits first
                attempts = [
                    (cand_up, cand_down) if flip_balls == 0 else (cand_down, cand_up),
                    (cand_left, cand_right) if flip_balls == 0 else (cand_right, cand_left),
                ]
            else:
                # Try horizontal splits first
                attempts = [
                    (cand_left, cand_right) if flip_balls == 0 else (cand_right, cand_left),
                    (cand_up, cand_down) if flip_balls == 0 else (cand_down, cand_up),
                ]

            applied = False
            for c1, c2 in attempts:
                if in_bounds(c1, self.size) and in_bounds(c2, self.size):
                    # must not collide with players or the other ball's prev pos
                    occ = {self.p1_pos, self.p2_pos, prev_b1, prev_b2}
                    if c1 not in occ and c2 not in occ:
                        intended_b1, intended_b2 = c1, c2
                        applied = True
                        break

            if not applied:
                # revert to previous positions for both
                intended_b1 = prev_b1
                intended_b2 = prev_b2
                b1_moves = b2_moves = False

        # If intended ball positions collide with players (not pushing)
        # We'll block ball movement into occupied player cells (ball stays).
        if b1_moves and (intended_b1 == self.p1_pos or intended_b1 == self.p2_pos):
            intended_b1 = prev_b1
            b1_moves = False
        if b2_moves and (intended_b2 == self.p1_pos or intended_b2 == self.p2_pos):
            intended_b2 = prev_b2
            b2_moves = False

        # Now apply players movement, considering pushes that succeeded
        # If a player attempted to move into a ball cell and that ball
        # successfully moved (b1_moves or b2_moves and intended pos != prev),
        # then player can move into the ball's previous pos.
        # If the ball didn't move, the player stays.
        final_p1 = self.p1_pos
        final_p2 = self.p2_pos

        # Player1 logic
        if p1_moves:
            wanted = add_tuples(self.p1_pos, dp1)
            # If wanted was a ball cell:
            if wanted == prev_b1:
                if b1_moves and intended_b1 != prev_b1:
                    # ball moved: p1 can move into previous ball position
                    final_p1 = wanted
                else:
                    # ball didn't move -> p1 stays
                    final_p1 = self.p1_pos
            elif wanted == prev_b2:
                if b2_moves and intended_b2 != prev_b2:
                    final_p1 = wanted
                else:
                    final_p1 = self.p1_pos
            else:
                # desired cell may be empty or occupied by other player (we handled same-target earlier)
                # Also prevent moving into opponent current pos (should have been resolved earlier)
                if wanted != self.p2_pos:
                    final_p1 = wanted
                else:
                    final_p1 = self.p1_pos
        else:
            final_p1 = self.p1_pos

        # Player2 logic
        if p2_moves:
            wanted = add_tuples(self.p2_pos, dp2)
            if wanted == prev_b1:
                if b1_moves and intended_b1 != prev_b1:
                    final_p2 = wanted
                else:
                    final_p2 = self.p2_pos
            elif wanted == prev_b2:
                if b2_moves and intended_b2 != prev_b2:
                    final_p2 = wanted
                else:
                    final_p2 = self.p2_pos
            else:
                if wanted != final_p1:  # prevent stepping into p1 new pos
                    final_p2 = wanted
                else:
                    # if they would step into p1's new position (which moved),
                    # then resolve 50/50 random earlier — but if gets here, block.
                    final_p2 = self.p2_pos
        else:
            final_p2 = self.p2_pos

        # Apply ball moves and handle falling off edges and scoring
        r1_gain = 0
        r2_gain = 0

        # b1
        if b1_moves:
            # Check fall/score
            new_b1, s1 = self._apply_bounds_and_score(intended_b1, ball_index=1)
            if s1:
                r1_gain += s1  # owner p1 gets point only for left edge falls
            if new_b1 is not None:
                self.b1_pos = new_b1
            # else _apply_bounds_and_score already respawned the ball
        else:
            # unchanged
            self.b1_pos = prev_b1

        # b2
        if b2_moves:
            new_b2, s2 = self._apply_bounds_and_score(intended_b2, ball_index=2)
            if s2:
                r2_gain += s2
            if new_b2 is not None:
                self.b2_pos = new_b2
        else:
            self.b2_pos = prev_b2

        # Update scores
        self.p1_score += r1_gain
        self.p2_score += r2_gain

        # ---------- Finalize player positions ----------
        self.p1_pos = final_p1
        self.p2_pos = final_p2

        # Ensure no overlap occurs (shouldn't): if overlap occurs, push player2 back to previous.
        if self.p1_pos == self.p2_pos:
            # revert player2
            self.p2_pos = prev_p2

        # If a player ended occupying a ball cell (shouldn't after pushes), fix by respawning the ball
        if self.p1_pos == self.b1_pos:
            # move ball to random empty
            self._respawn_ball(1)
        if self.p1_pos == self.b2_pos:
            self._respawn_ball(2)
        if self.p2_pos == self.b1_pos:
            self._respawn_ball(1)
        if self.p2_pos == self.b2_pos:
            self._respawn_ball(2)

        # Step done. Create observation and rewards
        obs = self._get_obs()
        rewards = np.array([r1_gain, r2_gain], dtype=np.float32)

        # Termination/truncation
        terminated = (self.p1_score >= self.goal_score) or (self.p2_score >= self.goal_score)
        truncated = self.step_count >= self.max_steps  # This was correct

        info = {
            "step_count": self.step_count,
            "p1_score": self.p1_score,
            "p2_score": self.p2_score,
            "prev_positions": {
                "p1": prev_p1,
                "p2": prev_p2,
                "b1": prev_b1,
                "b2": prev_b2,
            },
            "intended": {
                "p1": desired_p1,
                "p2": desired_p2,
                "b1": intended_b1,
                "b2": intended_b2,
            },
        }

        # Force episode end if max steps reached
        if truncated:
            return obs, rewards, False, True, info  # Early return when truncated

        return obs, rewards, bool(terminated), bool(truncated), info

    def render(self, mode: str = "human") -> None:
        """Render current game state.

        Args:
            mode: Rendering mode (only "human" supported)
        """
        # Create simple ASCII rendering
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]

        # Place balls first so players can override if they ended in same cell (shouldn't happen)
        def mark(pos, sym):
            r, c = pos
            if 0 <= r < self.size and 0 <= c < self.size:
                grid[r][c] = sym

        mark(self.b1_pos, "b1")
        mark(self.b2_pos, "b2")
        mark(self.p1_pos, "P1")
        mark(self.p2_pos, "P2")

        lines = []
        header = f"Step {self.step_count} | Scores -> P1: {self.p1_score} | P2: {self.p2_score}"
        lines.append(header)
        lines.append("-" * (5 * self.size))
        for r in range(self.size):
            row_elems = []
            for c in range(self.size):
                cell = grid[r][c]
                # pad to width 2 for alignment
                row_elems.append(cell.rjust(2))
            lines.append(" ".join(row_elems))
        lines.append("-" * (5 * self.size))
        out = "\n".join(lines)
        print(out)
