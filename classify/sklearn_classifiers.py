""" Classifiers based on the sklearn
    The all have the same interface, so the can be wrapped in one class
    Derived from TextClassifier
"""

from typing import List
from text_classifier import TextClassifier, ClassifierResult

import json
import os
import pandas

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

class SklearnClassifier(TextClassifier):
    """
    Classify with sklearn classifiers
    """
    supported_classifiers = ["RandomForestClassifier", "LogisticRegression", "MLPClassifier",
                               "GaussianNB", "MultinomialNB", "KNeighborsClassifier", "SVC", "Perceptron"]

    def __init__(self, classifier_type :str, model_folder_path:str = None , verbose = False):
        """
        Initialize the classifier
        :param classifier_type: The name of the classifiers
        :param model_folder_path:
        :param verbose:
        """
        if model_folder_path:
            self.model_folder_path = model_folder_path
        else:
            model_path = os.path.abspath(__file__)
            model_path = os.path.dirname(model_path)
            model_path = os.path.join(model_path, "data", "sklearn_classifier_models", classifier_type)
            self.model_folder_path = model_path
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        # Store the file path of the training data
        self.training_data = None
        self.verbose = verbose

        if classifier_type == "KNeighborsClassifier":
            self.sklearn_classifier = KNeighborsClassifier(n_jobs=3)
        elif classifier_type == "MLPClassifier":
            self.sklearn_classifier = MLPClassifier()
        elif classifier_type == "SVC":
            self.sklearn_classifier = SVC(kernel="linear", C=0.025)
        elif classifier_type == "GaussianNB":
            self.sklearn_classifier = GaussianNB()
        elif classifier_type == "MultinomialNB":
            self.sklearn_classifier = MultinomialNB(alpha=0.01)
        elif classifier_type == "LogisticRegression":
            self.sklearn_classifier = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
        elif classifier_type == "RandomForestClassifier":
            self.sklearn_classifier = RandomForestClassifier(n_estimators=10)
        elif classifier_type == "Perceptron":
            self.sklearn_classifier = Perceptron()
        else:
            raise Exception("Unsupported classifier type {0}. Use one of {1}".format(classifier_type, self.supported_classifiers))

        self.classifier_type = classifier_type
        self.count_vectorizer = None
        self.tfidf_transformer = None

    def set_count_vectorizer(self, count_vectorizer:CountVectorizer):
        """
        Set the count vectorizer (fit transform done during training)
        Configurable for evaluation purposes. If this is not called, a default vectorizer is used
        :param count_vectorizer: Vectorizer to replace the default one
        :return: None
        """
        self.count_vectorizer = count_vectorizer

    def set_tfidf_transformer(self, tfidf_transformer:TfidfTransformer):
        """
        Set the tfidf transformer (fit transform done during training)
        Configurable for evaluation purposes. If this is not called, a default transformer is used
        :param tfidf_transformer: Transformer to replace the default one
        :return: None
        """
        self.tfidf_transformer = tfidf_transformer

    def classify(self, data: dict, text_label: str) -> List[ClassifierResult]:
        """
        Classify a record consisting of text and sensor codes
        :return The detected class as ClassifierResult
        """
        # print(sensor_codes)
        data_point = {}
        data_point["text"] = data[text_label]
        data_to_classify = self._create_data_table([data_point])
        # print("\nData table: \n{0}\n".format(data_to_classify))
        matrix_counts = self.count_vectorizer.transform(data_to_classify.text)

        matrix_tf = self.tfidf_transformer.transform(matrix_counts)
        matrix_tf = matrix_tf.toarray()
        predicted = self.sklearn_classifier.predict(matrix_tf)
        predicted_class = predicted[0]

        result = ClassifierResult(predicted_class, -1, "")
        return [result]

    def train(self, training_data: str, text_label: str, class_label: str) -> None:
        """
        Train the classifier
        :param training_data: File name. Training data is one json per line
        :param text_label: Json field which contains the text
        :param class_label:  Json field which contains the label for the classes to train
        :return: Nothing
        """
        """
        Train the algorithm with the data from the knowledge graph
        """
        self.training_data = training_data
        data_train = self._create_data_table_from_training_file(training_data, text_label, class_label)
        data_train = data_train.fillna(0)
        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(min_df=10, max_df=0.8, ngram_range=(1, 1))
        matrix_train_counts = self.count_vectorizer.fit_transform(data_train.text)
        if self.tfidf_transformer is None:
            self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(matrix_train_counts)
        matrix_train_tf = self.tfidf_transformer.transform(matrix_train_counts)
        matrix_train_tf = matrix_train_tf.toarray()
        self.sklearn_classifier.fit(matrix_train_tf, data_train.label)


    # ********************************
    # Creation  of data to classify
    # ********************************
    def _create_data_table_from_training_file(self, training_file: str, text_label: str, class_label: str):
        """
        Create a data table (for training)
        """
        with open(training_file, encoding = 'utf-8') as training_fp:
            datapoints = []
            for line in training_fp:
                record = json.loads(line)
                datapoint = {}
                datapoint["text"] = record[text_label]
                datapoint["label"] = record[class_label]
                datapoints.append(datapoint)

        return self._create_data_table(datapoints)

    def _create_data_table(self, datapoints):
        # shuffle for good measure
        datapoints = shuffle(datapoints)
        data_table = pandas.DataFrame(datapoints)
        return data_table
