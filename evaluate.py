import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from environment import TrussEnv
from agent import DQNAgent
from anastruct import SystemElements

RESULTS_DIR = "results"
MODELS_DIR = "models"
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.csv")

def plot_training_curve():
    if not os.path.exists(LOG_PATH):
        print(f"Error: {LOG_PATH} not found.")
        return
    df = pd.read_csv(LOG_PATH)

    # plotting the final step score (quality) to align with the optimization goal
    target_col = 'Final_Step_Score'
    label_text = 'Step Score (Design Quality)'
    plt.figure(figsize=(10, 5))
    plt.plot(
        df['Episode'], df[target_col], label='Raw Score', alpha=0.3,
        color='gray'
        )

    # average for smooth line
    if len(df) > 20:
        rolling_mean = df[target_col].rolling(window=20).mean()
        plt.plot(
            df['Episode'], rolling_mean, label='Smoothed (20 ep avg)',
            color='blue', linewidth=2
            )

    plt.title(f"Agent Learning Curve ({label_text})")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = os.path.join(RESULTS_DIR, "plot_learning_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Saved: {save_path}")

def save_structure_plot(structure_vec, physics, filename, title):
    ss = SystemElements()

    # same pruning as the physics engine
    if hasattr(physics, '_get_cleaned_structure'):
        structure_vec = physics._get_cleaned_structure(structure_vec)

    # 1. rebuild structure from binary vector
    for i, exists in enumerate(structure_vec):
        if exists == 1:
            n1, n2 = physics.all_possible_bars[i]
            ss.add_element(
                location=[physics.nodes[n1], physics.nodes[n2]],
                spring={1: 0, 2: 0}
            )

    # 2. rebuild supports & loads
    fixed_locs = physics.config.get('fixed_supports', [])
    for supp_loc in fixed_locs:
        nid = ss.find_node_id(vertex=supp_loc)
        if nid:
            ss.add_support_hinged(node_id=nid)
    rolling_data = physics.config.get('rolling_supports', [])
    for item in rolling_data:
        loc = item['location']
        direction = item['direction']
        nid = ss.find_node_id(vertex=loc)
        if nid:
            ss.add_support_roll(node_id=nid, direction=direction)
    load_loc = physics.config['load_location']
    nid = ss.find_node_id(vertex=load_loc)
    if nid: ss.point_load(node_id=nid, Fy=physics.config['load_force_y'])

    # 3. solve & plot
    ss.solve(force_linear=True, verbose=False)
    save_path = os.path.join(RESULTS_DIR, filename)
    plt.figure()
    ss.show_axial_force(show=False, values_only=False)
    load_x, load_y = physics.config['load_location']
    force_val = physics.config['load_force_y']
    arrow_len = 0.5
    dx, dy = 0, -arrow_len
    plt.arrow(
        load_x, load_y, dx, dy,
        head_width=0.1, head_length=0.1,
        fc='red', ec='red', width=0.01, zorder=10,
        length_includes_head=True
        )
    plt.text(
        load_x + 0.1, load_y - 0.5, f"F={force_val}",
        color='red', fontsize=10
    )
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Image saved: {filename}")

def visualize_model(config_path, model_filename, output_filename, title):
    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.exists(model_path):
        print(f"Skipping {title}: {model_filename} not found.")
        return
    env = TrussEnv(config_path=config_path)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    agent.load(model_path)
    agent.epsilon = 0.0  # no random moves
    state, _ = env.reset()
    final_state = state.copy()
    for _ in range(50):
        action = agent.act(state)
        state, _, term, _, _ = env.step(action)
        if not term:
            final_state = state.copy()
        else:
            break
    mass = np.sum(final_state)
    full_title = f"{title}\n(Mass: {mass} | Config: {os.path.basename(config_path)})"
    save_structure_plot(final_state, env.physics, output_filename, full_title)

def run_benchmark(config_path):
    print("\n--- RUNNING BENCHMARK (Random vs RL) ---")

    # 1. environment setup
    env = TrussEnv(config_path=config_path)
    physics = env.physics
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    w_pen = physics.config.get('weight_penalty')
    s_pen = physics.config.get('stiffness_penalty')
    agent.load(os.path.join(MODELS_DIR, "best_model.keras"))
    agent.epsilon = 0.0

    # 2. data collection
    # random baseline (500 episodes)
    print("   ...Collecting Random Baseline (500 episodes)")
    random_data = []
    for _ in range(500):
        action = np.random.randint(0, 2, size=state_size)
        res = physics.solve(action)
        valid = res.valid
        mass = res.weight if valid else np.sum(action)  # Raw mass if invalid
        disp = res.max_displacement if valid else 10.0  # Cap disp if invalid
        if not valid:
            score = -50.0
        else:
            disp_cost = min(res.max_displacement * s_pen, 20.0)
            score = 10.0 - (res.weight * w_pen) - disp_cost
        random_data.append(
            {'valid': valid, 'mass': mass, 'disp': disp, 'score': score}
            )
    # RL agent (10 runs)
    print("   ...Testing RL Agent (10 runs)")
    agent_data = []
    for _ in range(10):
        state, _ = env.reset()
        final_score = -50.0
        final_struct = state.copy()
        for _ in range(50):
            action = agent.act(state)
            state, r, term, _, _ = env.step(action)
            final_struct = state.copy()
            if term:
                final_score = r if r > -40 else -50.0
                break
            final_score = r
        res = physics.solve(final_struct)
        valid = res.valid
        agent_data.append(
            {
                'valid': valid,
                'mass': res.weight if valid else np.sum(final_struct),
                'disp': res.max_displacement if valid else 10.0,
                'score': final_score
            }
        )

    # 3. Visualizations & Stats
    r_valid_data = [d for d in random_data if d['valid']]
    a_valid_data = [d for d in agent_data if d['valid']]
    r_success_rate = (len(r_valid_data) / len(random_data)) * 100
    a_success_rate = (len(a_valid_data) / len(agent_data)) * 100
    print(f"\n   --- RESULTS ---")
    print(f"   Random Success Rate: {r_success_rate:.1f}%")
    print(f"   RL Agent Success Rate: {a_success_rate:.1f}%")

    # plot 1: score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(
        [d['score'] for d in random_data], bins=30, color='gray', alpha=0.5,
        label='Random Search'
        )
    plt.hist(
        [d['score'] for d in agent_data], bins=10, color='red', alpha=0.7,
        label='RL Agent', edgecolor='red', linewidth=3.0
        )
    plt.title("Score Distribution: Random vs RL")
    plt.xlabel("Score")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "eval_1_score_dist.png"))
    plt.close()

    # plot 2: success rates
    plt.figure(figsize=(6, 6))
    bars = plt.bar(
        ['Random', 'RL Agent'], [r_success_rate, a_success_rate],
        color=['gray', 'red']
        )
    plt.ylim(0, 110)
    plt.ylabel("Success Rate (%)")
    plt.title("Reliability Check")
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}%",
            ha='center', fontweight='bold'
            )
    plt.savefig(os.path.join(RESULTS_DIR, "eval_2_success_rate.png"))
    plt.close()

    # plot 3: design space
    plt.figure(figsize=(10, 6))
    if r_valid_data: # random valid only
        plt.scatter(
            [d['mass'] for d in r_valid_data],
            [d['disp'] for d in r_valid_data],
            color='gray', alpha=0.5, label='Random (Valid)', s=30
        )
    if a_valid_data: # RL valid only
        plt.scatter(
            [d['mass'] for d in a_valid_data],
            [d['disp'] for d in a_valid_data],
            color='red', marker='*', s=200, label='RL Agent', zorder=10
        )
    plt.xlabel("Structure Mass (Cost)")
    plt.ylabel("Max Displacement (Stiffness)")
    plt.title("Design Space Efficiency (Lower-Left is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "eval_3_design_space.png"))
    plt.close()
    print(f"   -> Saved 3 benchmark plots to {RESULTS_DIR}")

def run_full_evaluation(config_path):
    print(f"\n--- STARTING EVALUATION ---")
    plot_training_curve()
    visualize_model(
        config_path, "best_model.keras", "result_best.png",
        "Best Performing Agent"
        )
    visualize_model(
        config_path, "final_model.keras", "result_final.png",
        "Final Converged Agent"
        )
    run_benchmark(config_path)