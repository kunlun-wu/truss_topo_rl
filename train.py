import os
import csv
from environment import TrussEnv
from agent import DQNAgent
from evaluate import save_structure_plot

MAX_STEPS = 50
BATCH_SIZE = 32
SAVE_INTERVAL = 500
RESULTS_DIR = "results"
MODELS_DIR = "models"
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def train_agent(config_path, episodes):
    print(f"\n--- STARTING TRAINING ---")
    print(f"Config: {config_path}")
    print(f"Max Steps: {MAX_STEPS}")
    print(
        f"Metric: Best Record tracked by FINAL STEP SCORE (Quality of resulting truss)"
        )
    env = TrussEnv(config_path=config_path)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    # snapshots
    snapshots_dir = os.path.join(RESULTS_DIR, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    snapshot_counter = 0
    # CSV Log Header
    with open(LOG_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Episode", "Total_Reward", "Final_Step_Score", "Final_Disp",
             "Final_Weight", "Epsilon"]
        )
    best_final_score = -float('inf')

    # episode loop
    for e in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        final_step_reward = -999
        final_info = {}
        episode_best_struct = None
        for time in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            final_step_reward = reward
            final_info = info
            if not terminated:
                episode_best_struct = state.copy()
            if len(agent.memory) > BATCH_SIZE:
                agent.replay()
            if terminated:
                break
        agent.decay_epsilon()
        agent.update_target_model()
        print(
            f"Ep: {e}/{episodes} | Total: {total_reward:.1f} | "
            f"Final Step: {final_step_reward:.2f} | "
            f"Disp: {final_info.get('displacement', 999):.4f} | "
            f"Mass: {final_info.get('weight', 0)} | "
            f"Eps: {agent.epsilon:.2f}"
        )
        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(
                [e, total_reward, final_step_reward,
                 final_info.get('displacement', 999),
                 final_info.get('weight', 0), agent.epsilon]
            )
        is_valid = final_info.get('valid', False)
        if is_valid and (final_step_reward > best_final_score):
            best_final_score = final_step_reward
            agent.model.save(os.path.join(MODELS_DIR, "best_model.keras"))
            print(f">>> 🏆 NEW BEST DESIGN: Step Score {best_final_score:.2f}")
            if episode_best_struct is not None:
                snapshot_counter += 1
                filename = os.path.join(
                    "snapshots", f"snapshot{snapshot_counter}.png"
                    )
                save_structure_plot(
                    episode_best_struct, env.physics,
                    filename,
                    f"Record #{snapshot_counter}: Final Step Score {best_final_score:.2f} (Mass {final_info['weight']})"
                )
        if e % SAVE_INTERVAL == 0:
            agent.model.save(os.path.join(MODELS_DIR, f"model_ep{e}.keras"))
    agent.model.save(os.path.join(MODELS_DIR, "final_model.keras"))
    print("--- ✅ TRAINING COMPLETE ---")