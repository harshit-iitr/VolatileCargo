from model import QuarterCarEnv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 2. TRAINING SCRIPT ---
if __name__ == "__main__":
    # Load Data
    try:
        df = pd.read_csv('data/road_profiles.csv')
        # Create a dictionary of {profile_name: numpy_array}
        road_data = {col: df[col].values for col in ['profile_1', 'profile_2', 'profile_3', 'profile_4', 'profile_5']}
        print("Data loaded for RL.")
    except FileNotFoundError:
        print("Error: road_profiles.csv not found.")
        exit()

    # Create Env
    env = DummyVecEnv([lambda: QuarterCarEnv(road_data)])

    # Initialize Agent (PPO is generally robust)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, ent_coef=0.01)

    print("Starting Training (This may take 5-10 mins)...")
    # Train for ~200,000 steps (roughly 50 loops over all profiles)
    model.learn(total_timesteps=200000)
    print("Training Complete!")
    
    # Save (Optional)
    model.save("stark_suspension_agent")

    # --- 3. GENERATE SUBMISSION ---
    print("\nGenerating Submission with Trained Agent...")
    submission_data = []
    
    # We define a separate calculation function for the exact metrics
    def calculate_metrics_final(zs_hist, acc_s_hist, dt=0.005):
        zs = np.array(zs_hist)
        acc = np.array(acc_s_hist)
        zs_rel = zs - zs[0]
        jerk = np.diff(acc, prepend=acc[0]) / dt
        rms_zs = np.sqrt(np.mean(zs_rel**2))
        max_zs = np.max(np.abs(zs_rel))
        rms_jerk = np.sqrt(np.mean(jerk**2))
        max_jerk = np.max(np.abs(jerk))
        score = (0.5 * rms_zs) + max_zs + (0.5 * rms_jerk) + max_jerk
        return score, rms_zs, max_zs, rms_jerk

    # Run Inference
    for p_name, r_data in road_data.items():
        # Setup specific env for testing
        test_env = QuarterCarEnv({p_name: r_data})
        obs, _ = test_env.reset()
        
        # Override the random choice in reset to force the current profile
        test_env.current_profile_data = r_data
        
        zs_hist = []
        acc_s_hist = []
        
        done = False
        while not done:
            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            
            # Step
            obs, reward, done, _, _ = test_env.step(action)
            
            # Record directly from state for accuracy
            zs_hist.append(test_env.state[0])
            acc_s_hist.append(obs[4]) # acc_s is index 4 in obs
            
        score, rms_zs, max_zs, rms_jerk = calculate_metrics_final(zs_hist, acc_s_hist)
        
        submission_data.append({
            'profile': p_name,
            'rms_zs': rms_zs,
            'max_zs': max_zs,
            'rms_jerk': rms_jerk,
            'comfort_score': score
        })
        print(f"{p_name}: Score = {score:.4f}")

    # Save
    sub_df = pd.DataFrame(submission_data)
    sub_df = sub_df[['profile', 'rms_zs', 'max_zs', 'rms_jerk', 'comfort_score']]
    sub_df.to_csv('submission_rl.csv', index=False)
    print("Done! Upload 'submission_rl.csv'.")