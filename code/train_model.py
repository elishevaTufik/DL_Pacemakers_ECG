#!/usr/bin/env python

import os, sys
from train_12ECG_classifier import train_12ECG_classifier

def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
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
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

        # Find files.
 

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print('Running training code...')
    ####לאזן את דאטה oversampling
    
    train_12ECG_classifier(input_directory, output_directory)
    

    
    print('Done.')


