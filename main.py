import glob
import os
from train import train_agent
from evaluate import run_full_evaluation

CONFIGS_DIR = "configs"

def select_config():
    files = glob.glob(os.path.join(CONFIGS_DIR, "*.json"))
    files.sort()
    print("\n--- Available Configurations ---")
    for i, f in enumerate(files):
        filename = os.path.basename(f)
        print(f"[{i + 1}] {filename}")
    while True:
        choice = input(
            "\nEnter the number of the config to use (or press Enter for default): "
            )
        if choice == "":
            return files[0]
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            return files[idx]
        else:
            print("Invalid number.")

if __name__ == "__main__":
    selected_config = select_config()
    ep_input = input("Enter number of episodes (default 500): ")
    episodes = int(ep_input) if ep_input.strip() else 500
    train_agent(selected_config, episodes)
    run_full_evaluation(selected_config)