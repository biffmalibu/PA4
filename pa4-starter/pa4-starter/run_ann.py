'''
File: run-ann.py
Author: Hank Feild
Date: 2024-04-19
Purpose: Provides an interface for running and evaluating a trained artificial neural network.
'''

import ann
import sys


def main():
    if(len(sys.argv) < 2 or '-h' in sys.argv or 
       not any([arg.startswith('-f=') for arg in sys.argv]) or
       not any([arg.startswith('-n=') for arg in sys.argv])):
        print('Usage: python run_ann.py [-h] [-f=<data file>] [-eval] [-header] [-n=<network input file>]',
              '',
              'Options:'
              '     -h: Print this help message and exit',
              '     -f: The file containing the data to run through the trained model;',
              '         should contain one row per observation, comma separated numeric values -- REQUIRED',
              '     -n: The file to load the network from -- REQUIRED',
              '     -header: Indicates that the data file has a header row, which will be removed (default: assumes no header)',
              '     -eval: if present, the data will be evaluated using the last column of the data file',
              '            as the true labels; if absent, predictions will be printed one per line in the',
              '            order of the data file',
              sep='\n')
        sys.exit(1)
        
    # Defaults
    data_file = None
    network_file = None
    skip_header = False
    eval_mode = False

    # Parse command line arguments
    for arg in sys.argv:
        if arg.startswith('-f='):
            data_file = arg[3:]
        elif arg.startswith('-n='):
            network_file = arg[3:]
        elif arg == '-header':
            skip_header = True
        elif arg == '-eval':
            eval_mode = True
    
    network = ann.ArtificalNeuralNetwork()
    network_data = network.load(network_file)

    if eval_mode:
        print('Evaluating network...')
        # Load data.
        # TODO Advanced 5: Implement the code to parse the test data and evaluate it 
        # with the network. This should report the accuracy.
    else:
        # Load data.
        int_to_label_lookup = {v:k for k, v in network.label_to_int_lookup.items()}
        dataset = network.load_and_process_data(data_file, skip_header, data_has_labels=False)
        predictions = network.predict(dataset)
        for prediction in predictions:
            print(int_to_label_lookup[prediction])


if __name__ == '__main__':
    main()