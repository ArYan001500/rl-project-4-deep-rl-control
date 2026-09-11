from AIUT_Environments.AIUT_Environments import MountainCar
import inspect

env = MountainCar()

print("reset signature:", inspect.signature(env.reset))
print("step signature:", inspect.signature(env.step))

print("\nInitial attributes before reset:")
print("p =", env.p)
print("v =", env.v)
print("stepsLeft =", env.stepsLeft)
print("terminated =", env.terminated)
print("truncated =", env.truncated)

reset_result = env.reset()

print("\nAfter reset:")
print("reset result:", reset_result)
print("p =", env.p)
print("v =", env.v)
print("stepsLeft =", env.stepsLeft)
print("terminated =", env.terminated)
print("truncated =", env.truncated)

print("\nTesting step actions:")
for action in [0, 1, 2]:
    env.reset()
    result = env.step(action)
    print("\naction =", action)
    print("step result:", result)
    print("p =", env.p)
    print("v =", env.v)
    print("stepsLeft =", env.stepsLeft)
    print("terminated =", env.terminated)
    print("truncated =", env.truncated)