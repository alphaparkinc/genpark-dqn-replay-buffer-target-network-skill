# genpark-dqn-replay-buffer-target-network-skill

Agent Skill implementing the **DQN Experience Replay Buffer and Polyak Soft Target Network Averaging**, stabilizing off-policy temporal difference reinforcement learning.

## Architectural Overview
```mermaid
flowchart TD
    Env["Environment Steps (s, a, r, s', done)"] --> Buffer["Circular Experience Replay Buffer"]
    Buffer --> Sample["Uniform Random Mini-batch Sampling"]
    Sample --> TargetCalc["Bellman Target via Target Network: y = r + gamma * max Q_target(s', a')"]
    TargetCalc --> Loss["TD Error Loss: (y - Q(s, a))^2"]
    Loss --> Online["Update Online Q-Network"]
    Online --> Polyak["Polyak Soft Update: Q_target = tau * Q + (1 - tau) * Q_target"]
```
