import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_history(csv_path="training_log.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found. Make sure the script is in the same directory as the CSV.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    required_cols = ['iteration', 'loss', 'mean_turn_reward', 'mean_episode_return']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: '{col}' column missing from the CSV.")
            return

    os.makedirs("plots", exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')


    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['loss'], color='tab:red', linewidth=1.5, alpha=0.85)
    plt.title('Training Loss over Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/loss_plot.png', dpi=300)
    plt.close()
    print("Successfully saved: plots/loss_plot.png")

    plt.figure(figsize=(10, 6))
    plt.plot(df['iteration'], df['mean_turn_reward'], color='tab:blue', linewidth=1.5, alpha=0.85)
    plt.title('Mean Turn Reward over Iterations', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Mean Turn Reward', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/turn_reward_plot.png', dpi=300)
    plt.close()
    print("Successfully saved: plots/turn_reward_plot.png")

    df_return = df.dropna(subset=['mean_episode_return'])
    
    plt.figure(figsize=(10, 6))
    if not df_return.empty:
        # We use markers here because episode returns might be spaced out
        plt.plot(df_return['iteration'], df_return['mean_episode_return'], 
                 color='tab:green', linewidth=1.5, marker='o', markersize=4, alpha=0.85)
        plt.title('Mean Episode Return over Iterations', fontsize=14, fontweight='bold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Episode Return / Seat', fontsize=12)
        plt.tight_layout()
        plt.savefig('plots/episode_return_plot.png', dpi=300)
        print("Successfully saved: plots/episode_return_plot.png")
    else:
        print("Notice: No completed episodes recorded yet to plot 'Mean Episode Return'.")
    plt.close()

if __name__ == "__main__":
    plot_training_history()