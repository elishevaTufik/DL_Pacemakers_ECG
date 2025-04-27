import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Load the DataFrame from CSV
df = pd.read_csv("under_sampling.csv")

# Initialize StratifiedKFold
n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create a new column for folds
df['fold'] = -1  # Initialize with -1 or another placeholder

# Assign folds
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['426783006'])):
    df.loc[val_idx, 'fold'] = fold

# Save the resulting DataFrame to a new CSV file
df.to_csv("underSampling_with_folds.csv", index=False)