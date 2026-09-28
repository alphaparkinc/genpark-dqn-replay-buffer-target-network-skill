from client import DQNReplayTarget

dqn = DQNReplayTarget(capacity=50, tau=0.2)
for i in range(10):
    dqn.push(s=0, a=1, r=5.0, s_next=1, done=False)

loss = dqn.update_batch(batch_size=4)
print(f"Batch Mean TD Loss: {loss:.4f}")
print("Online Q-Values:", dqn.q_table[0])
print("Target Q-Values:", dqn.target_q_table[0])
