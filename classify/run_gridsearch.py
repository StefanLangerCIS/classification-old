"""
Evaluate any of the classifiers, print a confusion matrix and create further evalution metrics
"""
import argparse
from datetime import datetime
import json
import os
import sys
from sklearn_gridsearch import SklearnGridSearch
import sklearn.metrics
import sklearn.exceptions
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time
import numpy as np
import matplotlib.pyplot as plt

import warnings


def main():
    parser = argparse.ArgumentParser(description='Evaluate one or several text classifiers')

    # All available classifier types
    classifier_types = SklearnGridSearch.supported_classifiers

    parser.add_argument('--training',
                        default= r"C:\ProjectData\Uni\ltrs\classifier\classifier_data_train.json",
                        help='The training data for the classifier. If "None", the existing model is loaded (if it exists)')
    
    parser.add_argument('--input',
                        default= r"C:\ProjectData\Uni\ltrs\classifier\classifier_data_eval.json",
                        help='The text data to use for evaluation (one json per line)')

    parser.add_argument('--output',
                        default= r"C:\ProjectData\Uni\classif\classification-results",
                        help='Folder where to write the evaluation results')

    parser.add_argument('--classifier',
                        choices=classifier_types + ["all"],
                        default="RandomForestClassifier",
                        help="The classifier to use. If 'all' iterate through all available classifiers" )

    parser.add_argument('--text_label',
                        default="text",
                        help='Label/field in the json data contains the text to classify')

    parser.add_argument('--label',
                        default="author",
                        help='Label/field to use for training and classification')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Provide verbose output')

    args = parser.parse_args()

    # Run all classifiers
    if args.classifier == "all":
        classifiers = classifier_types
    else:
        classifiers = [args.classifier]

    print("INFO: Evaluating classifier(s) hyperparameters {0}".format(classifiers))


    # Iterate over the classifiers
    for classifier_type in classifiers:
        parameter_grid = {
                        "criterion" : ["gini", "entropy"],
                        "max_depth" : [5, 10, 20, 40, 100, 200],
                        "min_samples_leaf" : [1, 2, 4, 8]
                        }

        grid_search = SklearnGridSearch(classifier_type, parameter_grid)
        print("INFO: grid evaluating {0}".format(classifier_type))
        res = grid_search.grid_search(args.training, args.text_label, args.label)
        outfile_name = os.path.join(args.output, "gridsearch_results_{0}.txt".format(classifier_type))
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write("Classifier: {0}\n".format(classifier_type))
            for key in res.keys():
                outfile.write("Parameter {}: {}\n".format(key, res[key]))

if __name__ == "__main__":
    main()