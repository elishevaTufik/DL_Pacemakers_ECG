import pandas as pd


if __name__ == '__main__':

    # input_directory = sys.argv[1]
    # output_directory = sys.argv[2]
 

   # Load the DataFrame from the CSV file
    df = pd.read_csv('records_stratified_10_folds_v2.csv')

    # Specify the columns you want to keep
    columns_to_keep = ['Unnamed: 0','Patient', '426783006', '10370003']

    # Create a new DataFrame with only the specified columns
    df_filtered = df[columns_to_keep]

    # Optionally, save the new DataFrame to a new CSV file
    df_filtered.to_csv('filtered_file_07_10.csv', index=False)
    print('Running training code...')
  
    
    print('Done.')