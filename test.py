from environment import TrussEnv
from main import select_config
# test file to sanity check if the starting structure is valid
selected_config = select_config()
env = TrussEnv(selected_config)
env.reset()
result = env.physics.solve(env.current_structure)
print(f"Start Valid: {result.valid}")
print(f"Start Displacement: {result.max_displacement}")
print(f"Start Weight: {result.weight}")