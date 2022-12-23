# hotel-reviews-classification
Developed perceptron classifiers to identify hotel reviews as either truthful or deceptive, and either positive or negative.

## Perceptron Learning
- Contains logic for preprocessing hotel reviews data and single layer perceptron (vanilla and averaged models).
- The argument is a single file containing the training data; the program learns perceptron models, and writes the model parameters to two files: vanillamodel.txt for the vanilla perceptron, and averagedmodel.txt for the averaged perceptron.

## Perceptron Classification
- Contains logic for predicting on unseen hotel reviews.
- The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to a file containing the test data; the program reads the parameters of a perceptron model from the model file, classifies each entry in the test data, and writes the results to a text file called percepoutput.txt.

## Execution

- The learning program will be invoked in the following way:
`` python perceplearn.py /path/to/input ``

- The classification program will be invoked in the following way:
`` python percepclassify.py /path/to/model /path/to/input ``