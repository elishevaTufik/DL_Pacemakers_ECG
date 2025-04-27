import sys
import os
import pandas as pd
from imblearn.over_sampling import RandomOverSampler


def overSampling(df):
    # Load your data
    # df = pd.read_csv('new_record_2000.csv')
    print("1")
    # Separate features and target
    X = df.drop(columns=['10370003'])
    y = df['10370003']
    print("2")
    # Apply RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    print("3")
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print("4")
    # Create a new DataFrame with resampled data
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['10370003'] = y_resampled

    df_resampled.to_csv("train_over_sampling.csv",index=False)
    print(sum(df_resampled['10370003']==1))
    return df_resampled
    




