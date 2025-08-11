# Two-Player Push Environment

A custom Gymnasium environment for reinforcement learning research implementing a competitive two-player pushing game.

## 🎮 Game Description

In this environment, two players compete to push balls off the opposite edges of an 8x8 grid. The game features:

- **Two players**: Each can move up, down, left, right or stay still
- **Two balls**: Can be pushed by either player
- **Scoring**:
  - Player 1 scores when a ball falls off the left edge
  - Player 2 scores when a ball falls off the right edge
  - Any ball falling off top/bottom edges respawns randomly
- **Collisions**:
  - Players cannot occupy the same cell
  - When balls collide, they randomly split vertically or horizontally
  - Objects cannot pass through each other

## 🔧 Installation

```bash
git clone https://github.com/CarloCDT/two-player-push-env.git
cd two-player-push-env
pip install -e .
```

## 🚀 Usage

```python
from env import TwoPlayerPushEnv

# Create environment
env = TwoPlayerPushEnv()

# Reset to start new episode
obs = env.reset()

# Take action (both players)
# Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
action = [4, 3]  # P1 moves right, P2 moves left
obs, rewards, terminated, truncated, info = env.step(action)

# Render the game state
env.render()
```

## 🎯 Observation & Action Spaces

### Observation Space
- Type: `Box(8, 8, 4)`
- 4 binary channels representing:
  1. Player 1 position
  2. Player 2 position
  3. Ball 1 position
  4. Ball 2 position

### Action Space
- Type: `MultiDiscrete([5, 5])`
- Each player chooses from 5 actions:
  - 0: Stay
  - 1: Move Up
  - 2: Move Down
  - 3: Move Left
  - 4: Move Right

## 🏆 Rewards

- +1 point when a ball falls off opponent's edge
- 0 points otherwise
- Game ends when either player reaches 20 points or after 100 steps

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
