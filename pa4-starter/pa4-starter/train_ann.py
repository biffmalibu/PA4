'''
File: train-ann.py
Author: Hank Feild
Date: 2024-04-19
Purpose: Provides an interface for training and evaluating an artificial neural network.
'''

import sys
import ann


def main():
    if(len(sys.argv) < 2 or '-h' in sys.argv or not any([arg.startswith('-f=') for arg in sys.argv])):
        print('Usage: python train_ann.py [-h] [-f=<training file>] [-t=<test file>] [-header]',
              '     [-cv=<folds>] [-lr=<learning rate>] [-e=<epochs>] [-hl=<hidden layers>]',
              '     [-hln=<nodes per hidden layer>] [-n=<network output file>] [-scale] [-ern=<epochs to report>]',
              '',
              'Options:'
              '     -h: Print this help message and exit',
              '     -f: The file containing the training or CV data; should contain one row per ',
              '         observation, comma separated numeric values; last column is the class label-- REQUIRED',
              '     -t: The file containing the testing data (same format as -f); **using this turns',
              '         CV off even if -cv is specified**',
              '     -header: Indicates that the data file has a header row, which will be removed',
              '              (default: assumes no header)',
              '     -cv: The number of cross-validation folds (CV with n=10 will be used if ommitted',
              '          and -t is not specified)',
              '     -lr: The learning rate (default=0.3)',
              '     -e: The number of epochs (default=500)',
              '     -hl: The number of hidden layers (default=1)',
              '     -hln: The number of nodes per hidden layer (default=5)',
              '     -n: The file to write the network to -- this turns off CV',
              '     -scale: Scale the attributes (default: raw attributes used as features)',
              '     -ern: Ever # epochs during training, the accuracy of the network on the training data will be reported (default=50)',
              sep='\n')
        sys.exit(1)
        
    # Defaults
    train_file = None
    test_file = None
    skip_header = False
    cv_folds = 5
    network_file = None

    network = ann.ArtificalNeuralNetwork()

    
    # Parse command line arguments
    for arg in sys.argv:
        if arg.startswith('-f='):
            train_file = arg[3:]
            test_file = arg[3:]
        elif arg == '-header':
            skip_header = True
        elif arg.startswith('-cv='):
            cv_folds = int(arg[4:])
        elif arg.startswith('-ern='):
            network.epoch_report_n = int(arg[5:])
        # TODO Basic 4: read in the other command line arguments except for -n; 
        # store them in the correct variables (either local variables or in the
        # network object; use context to assist you).
        # TODO Advanced 3: read in the -n option (same rules as TODO Basic 4).

    print('Settings:',
          f'    Training file: {train_file}',
          f'    Testing file:{test_file}' if test_file else f'    CV: {cv_folds} folds',
          f'    Skip header: {skip_header}',
          f'    Learning rate: {network.learning_rate}',
          f'    Epochs: {network.epochs}',
          f'    Hidden layers: {network.hidden_layers}',
          f'    Nodes per hidden layer: {network.nodes_per_hidden_layer}',
          f'    Network file: {network_file}',
          f'    Scale features: {network.scale_features}',
          f'    Epoch report number: {network.epoch_report_n}',
          sep='\n')
    
    # Load training data.
    train_data = network.load_and_process_data(train_file, skip_header)


    # TODO Basic 5: if a test file is provided, train on the training data and test on the test data
    # instead of doing cross-validation. Report the accuracy of the test data.

    # TODO Advanced 4: Update this to also avoid CV if a network file is provided; and make sure to save the network to the file.

    # Run cross-validation on the one data file.
    accuracies, meanAccuracy = network.cross_validate(train_data, cv_folds)
    print(f'Accuracy by fold: {accuracies}%')
    print(f'Mean accuracy: {meanAccuracy}%')

if __name__ == '__main__':
    main()