from AIUT_Environments.AIUT_Environments import MountainCar
import inspect

print("MountainCar class:", MountainCar)
print("Constructor:", inspect.signature(MountainCar))

env = MountainCar()

print("\nAvailable methods/attributes:")
for name in dir(env):
    if not name.startswith("_"):
        print(name)