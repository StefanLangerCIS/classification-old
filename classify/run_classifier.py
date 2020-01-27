"""
Evaluate any of the classifiers, print a confusion matrix and create further evalution metrics
"""
import argparse
from datetime import datetime
import json
import os
import sys
from sklearn_classifiers import SklearnClassifier
import sklearn.metrics
import sklearn.exceptions
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import time
import numpy as np
import matplotlib.pyplot as plt

import warnings

def plot_and_store_confusion_matrix(y_true: list,
                                    y_pred:list,
                                    file_name: str,
                                    normalize=False,
                                    cmap=plt.cm.Blues,
                                    show = False):
    """
    This function prints and plots the confusion matrix, and saves it to a file
    :param y_true: The true classes
    :param y_pred: The predicted classes
    :param file_name: The file name to store the image of the confusion matrix
    :param normalize: normalize numbers (counts to relative counts)
    :param cmap: Layout
    :param show: Display the matrix. If false, only store it
    :return: Nothing
    """
    np.set_printoptions(precision=2)
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(file_name)
    if(show):
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate one or several text classifiers')

    # All available classifier types
    classifier_types = SklearnClassifier.supported_classifiers

    parser.add_argument('--training',
                    default= r"D:\ProjectData\Uni\ltrs\classifier\classifier_data_train.json",
                    help='The training data for the classifier. If "None", the existing model is loaded (if it exists)')
    
    parser.add_argument('--input',
                    default= r"D:\ProjectData\Uni\ltrs\classifier\classifier_data_eval.json",
                    help='The text data to use for evaluation (one json per line)')

    parser.add_argument('--output',
                        default= r"D:\ProjectData\Uni\classification-results",
                        help='Folder where to write the classifier evaluation results')

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

    print("INFO: Evaluating classifier(s) {0}".format(classifiers))

    # Determine the number of training lines (for the record)
    n_training_lines = 0
    if args.training is not None:
        with open(args.training, encoding="utf-8") as training:
            for line in training:
                n_training_lines += 1

    # Iterate over the classifiers
    for classifier_type in classifiers:
        classifier = SklearnClassifier(classifier_type)
        print("INFO: Evaluating classification of classifier {0}".format(classifier_type))
        training_time = 0
        if args.training is not None:
            training_time = time.time()
            print("INFO: Training classifier")
            classifier.train(args.training, args.text_label, args.label)
            print("INFO: Training completed")
            training_time = int(time.time()-training_time)
        else:
            print("INFO: Using pre-trained classifier")

        classifier.verbose = args.verbose

        print("INFO: Starting classification of data in {0} with classifier {1}".format(args.input, classifier_type))
        predicted_classes = []
        expected_classes = []
        # Keep track of time used
        classification_time = time.time()
        with open(args.input, encoding="utf-8") as infile:
            for line in infile:
                json_data = json.loads(line)
                res = classifier.classify(json_data, args.text_label)
                class_name = "none"
                if len(res) > 0:
                    class_name = res[0].class_name
                predicted_classes.append(class_name)
                expected_classes.append(json_data[args.label])

        classification_time = int(time.time()-classification_time)
        print("INFO: Classification completed for classifier {0}".format(classifier_type))

        outfile_name = os.path.join(args.output, "results_{0}.txt".format(classifier_type))
        with open(outfile_name, "w", encoding="utf-8") as outfile:
            outfile.write("#Info:\n")
            outfile.write("Classifier: {0}\n".format(classifier_type))
            outfile.write("Label: {0}\n".format(args.label))
            outfile.write("\n#Counts:\n")
            outfile.write("Number of training data_records: {0}\n".format(n_training_lines))
            outfile.write("Number of classified data_records: {0}\n".format(len(expected_classes)))
            outfile.write("Number of unique classes in data_records: {0}\n".format(len(set(expected_classes))))
            outfile.write("Number of unique classes found: {0}\n".format(len(set(predicted_classes))))
            outfile.write("\n#Performance:\n")
            outfile.write("Seconds used for training: {0}\n".format(training_time))
            outfile.write("Seconds used for classification: {0}\n".format(classification_time))

            warnings.filterwarnings("ignore", category = sklearn.exceptions.UndefinedMetricWarning)
            outfile.write("\n#Classification report:\n{0}\n".format(sklearn.metrics.classification_report(expected_classes, predicted_classes)))

            outfile.write("\n#Confusion matrix:\n{0}\n".format(
                sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)))

        # Also store confusion matrix as image
        imagefile_name = os.path.join(args.output, "results_{0}.jpg".format(classifier_type))
        plot_and_store_confusion_matrix(expected_classes, predicted_classes, imagefile_name)


if __name__ == "__main__":
    main()