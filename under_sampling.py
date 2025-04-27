import sys
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def balanc():
    

    # Load your data
    df = pd.read_csv('filtered_file_07_10.csv')
    # Separate features and target
    X = df.drop(columns=['10370003'])
    y = df['10370003']
    # Apply RandomOverSampler
    ros = RandomUnderSampler(sampling_strategy={0: 592, 1: 296})
    X_resampled, y_resampled = ros.fit_resample(X, y)
    # Create a new DataFrame with resampled data
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['10370003'] = y_resampled
   
    print("df_resampled['10370003'].value_counts():")
    df_resampled.to_csv("under_sampling.csv",index=False)
    print(sum(df_resampled['10370003']==1))
    



if __name__ == '__main__':



    balanc()
    print('Running training code...')
  
    
    print('Done.')