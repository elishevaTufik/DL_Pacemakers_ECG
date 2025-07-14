# #!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier

def load_challenge_data(filename):
    # Load the .npy file
    x = np.load(filename)
    
    # Transpose if the shape is (5000, 12) instead of (12, number_of_samples)
    if x.shape == (5000, 12):
        x = x.T  # Transpose to (12, 5000)
    
    # Ensure the data is in the correct format
    if x.shape[0] != 12:
        raise ValueError(f"Unexpected shape {x.shape}. Expected (12, number_of_samples).")
    
    # Convert to float64 for model compatibility
    data = np.asarray(x, dtype=np.float64)
    
    return data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.npy','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(class_string + '\n' + label_string + '\n' + score_string + '\n')


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')
    model_input = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):#running on the data
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('npy'):
            input_files.append(f)

    # print("input_files:")
    # for value in input_files:
    #     print("value",value)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model(model_input)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)
    print("num_files",num_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory,f)
        data = load_challenge_data(tmp_input_file)
        current_label, current_score,classes = run_12ECG_classifier(data, model)
        
        # Save results.
        save_challenge_predictions(output_directory,f,current_score,current_label,classes)


    print('Done.')