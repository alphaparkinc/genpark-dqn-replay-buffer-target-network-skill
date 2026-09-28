"""DQN Experience Replay & Polyak Target Network Engine.
100% Python Standard Library.
"""

import random

class DQNReplayTarget:
    """Deep Q-Network Experience Replay Buffer and Polyak Averaging Target Model."""
    def __init__(self, capacity=100, tau=0.1, gamma=0.95):
        self.capacity = capacity
        self.buffer = []
        self.tau = tau
        self.gamma = gamma
        self.q_table = [[0.0, 0.0] for _ in range(4)]
        self.target_q_table = [[0.0, 0.0] for _ in range(4)]

    def push(self, s, a, r, s_next, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, r, s_next, done))

    def update_batch(self, batch_size=4, lr=0.1):
        if len(self.buffer) < batch_size:
            return 0.0
        batch = random.sample(self.buffer, batch_size)
        total_td_err = 0.0

        for s, a, r, s_next, done in batch:
            target = r if done else r + self.gamma * max(self.target_q_table[s_next])
            td_err = target - self.q_table[s][a]
            self.q_table[s][a] += lr * td_err
            total_td_err += abs(td_err)

        for s in range(len(self.q_table)):
            for a in range(len(self.q_table[s])):
                self.target_q_table[s][a] = self.tau * self.q_table[s][a] + (1 - self.tau) * self.target_q_table[s][a]

        return total_td_err / batch_size
