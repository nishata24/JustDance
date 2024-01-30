import pandas as pd
import numpy as np
import math

def load_angles(file_path):
    return pd.read_csv(file_path)

def calculate_mse(file1, file2, penalty=0.25 * (180 ** 2)):
    """
    Calculate the Mean Square Error between angles in two files, with a penalty for missing joints.
    """
    mse = 0
    count = 0
    mse_values = []  # List to store individual MSE values

    # Determine the minimum length to avoid 'out-of-bounds' error
    min_length = min(len(file1), len(file2))

    for column in file1.columns:
        if column == 'timestamp':
            continue  # Skip timestamp column

        for i in range(min_length):
            angle1 = file1[column].iloc[i]
            angle2 = file2[column].iloc[i]
            
            mse_value = np.square(np.subtract(angle1, angle2)).mean()
            mse += mse_value
            
            if np.isnan(angle1) or np.isnan(angle2):
                mse += penalty  # Apply penalty for missing joints
            
            mse_values.append(mse_value)
            count += 1

    # Save MSE values to a CSV file
    mse_df = pd.DataFrame({'MSE': mse_values})
    mse_df.to_csv('mse_values.csv', index=False)

    return mse / count if count > 0 else 0

def compare_poses(file_path1, file_path2, alpha=0.0005):
    """
    Compare two poses and determine if they are the same.
    """
    angles_file1 = load_angles(file_path1)
    angles_file2 = load_angles(file_path2)

    mse = calculate_mse(angles_file1, angles_file2)
    score = 1 - math.tanh(alpha * mse)
    print(f"Score: {score}")

    return "Same Pose" if score >= 0.5 else "Different Pose"

# Example usage
file_path_1 = '/Users/nishatahmed/JustDance/angle_move1.csv'  # Replace with actual paths
file_path_2 = '/Users/nishatahmed/JustDance/angle_move1real.csv'

result = compare_poses(file_path_1, file_path_2)
print(f"Result: {result}")
