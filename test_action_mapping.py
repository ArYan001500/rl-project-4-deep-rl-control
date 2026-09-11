from AIUT_Environments.AIUT_Environments import MountainCar

env = MountainCar()

# DQN خروجی‌هایی با اندیس 0، 1، 2 می‌دهد
# ولی محیط AIUT اکشن‌های واقعی -1، 0، 1 می‌خواهد
ACTION_MAP = [-1, 0, 1]

state = env.reset()
print("Initial state:", state)

for action_index in [0, 1, 2]:
    env.reset()

    real_action = ACTION_MAP[action_index]
    next_state, reward, terminated, truncated = env.step(real_action)

    print("\nDQN action index:", action_index)
    print("Real AIUT action:", real_action)
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)