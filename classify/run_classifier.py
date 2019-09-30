"""
Run any of the classifier
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
import numpy as np
import matplotlib.pyplot as plt

import warnings

def plot_and_store_confusion_matrix(y_true, y_pred,
                          file_name,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix, and saves it to a file
    :param y_true: The true classes
    :param y_pred: The predicted clsases
    :param file_name: The file name to store the image of the confusion matrix
    :param normalize: normalize numbers (counts to relative counts)
    :param cmap: Layout
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
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Classify based on a text classifier')

    parser.add_argument('--training',
                    default= r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_training_data.json",
                    help='The training data for the classifier. If "None", the existing model is loaded')
    
    parser.add_argument('--input',
                    default= r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_test_data.json",
                    help='The texts to use for evaluation')

    parser.add_argument('--output',
                        default= r"D:\ProjectData\Uni\ltrs\data\classifier\results",
                        help='Folder where to write the classification results')

    parser.add_argument('--classifier',
                    choices=['RandomForestClassifier', 'MLPClassifier', 'KNeighborsClassifier', 'GaussianNB', 'MultinomialNB', 'SVC', 'LogisticRegression'],
                    default="RandomForestClassifier",
                        help='The classifier to use')

    parser.add_argument('--label',
                    default="author",
                    help='Label to use for training and classification')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Provide verbose output')

    args = parser.parse_args()

    print("Starting classifier {0} on {1}".format(args.classifier, args.input))

    if args.classifier == 'RandomForestClassifier':
        classifier = SklearnClassifier("RandomForestClassifier")
    elif args.classifier == 'KNeighborsClassifier':
        classifier = SklearnClassifier("KNeighborsClassifier")
    elif args.classifier == 'MLPClassifier':
        classifier = SklearnClassifier("MLPClassifier")
    elif args.classifier == 'GaussianNB':
        classifier = SklearnClassifier("GaussianNB")
    elif args.classifier == 'MultinomialNB':
        classifier = SklearnClassifier("MultinomialNB")
    elif args.classifier == 'SVC':
        classifier = SklearnClassifier("SVC")
    elif args.classifier == 'LogisticRegression':
        classifier = SklearnClassifier("LogisticRegression")

    n_training_lines = 0
    if args.training is not None:
        print("Training classifier")
        with open(args.training, encoding="utf-8") as training:
            for line in training:
                n_training_lines += 1
        classifier.train(args.training, args.label)
        print("Training completed")
    else:
        print("Using pre-trained classifier")

    classifier.verbose = args.verbose

    print("Starting classification of data in {0}".format(args.input))
    predicted_classes = []
    expected_classes = []
    with open(args.input, encoding="utf-8") as infile:
        for line in infile:
            json_data = json.loads(line)
            res = classifier.classify(json_data["text"])
            class_name = "none"
            if len(res) > 0:
                class_name = res[0].class_name
            predicted_classes.append(class_name)
            expected_classes.append(json_data[args.label])

    print("Classification completed.")

    outfile_name = os.path.join(args.output, "results_{0}.txt".format(args.classifier))
    with open(outfile_name, "w", encoding="utf-8") as outfile:
        outfile.write("Classifier: {0}\n".format(args.classifier))
        outfile.write("\n\nSTATISTICS\n\n")
        acc = sklearn.metrics.accuracy_score(expected_classes, predicted_classes)
        f1 = sklearn.metrics.f1_score(expected_classes, predicted_classes, average = 'micro')
        prec  = sklearn.metrics.precision_score(expected_classes, predicted_classes, average = 'micro')
        rec  = sklearn.metrics.recall_score(expected_classes, predicted_classes, average='micro')

        outfile.write("Overall:\n")
        outfile.write("Number of training records: {0}\n".format(n_training_lines))
        outfile.write("Number of classified records: {0}\n".format(len(expected_classes)))
        outfile.write("Number of unique classes in records: {0}\n".format(len(set(expected_classes))))
        outfile.write("Number of unique classes found: {0}\n".format(len(set(predicted_classes))))
        outfile.write("Accuracy: {0:.2f}\n".format(acc))
        outfile.write("F1 (micro): {0:.2f}\n".format(f1))
        outfile.write("Recall (micro): {0:.2f}\n".format(rec))
        outfile.write("Precision (micro): {0:.2f}\n".format(prec))

        warnings.filterwarnings("ignore", category = sklearn.exceptions.UndefinedMetricWarning)
        outfile.write("Classification report:\n{0}\n".format(sklearn.metrics.classification_report(expected_classes, predicted_classes)))

        outfile.write("Confusion matrix:\n{0}\n".format(
            sklearn.metrics.confusion_matrix(expected_classes, predicted_classes)))

    # Also store confusion matrix as image
    imagefile_name = os.path.join(args.output, "results_{0}.jpg".format(args.classifier))
    plot_and_store_confusion_matrix(expected_classes, predicted_classes, imagefile_name)

            
if __name__ == "__main__":
    main()