""" Classifiers based on the sklearn
    The all have the same interface, so the can be wrapped in one class
    Derived from TextClassifier
"""

from typing import List
from text_classifier import TextClassifier, ClassifierResult

import argparse
import glob
import json
import os
import pandas
import re

from scipy.sparse import csr_matrix, hstack

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression

class SklearnClassifier(TextClassifier):
    """
    Classifier based on ADS classifier
    """
    def __init__(self, classifier_type, model_folder_path = None , verbose = False):
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
            self.sklearn_classifier = KNeighborsClassifier()
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

        self.classifier_type = classifier_type
        self._load_models()
        self.set_hyperparameters()

    def set_hyperparameters(self, vectorizer_min_df=10, vectorizer_max_df=0.8):
        """
        Set the hyperparameters for the classifier
        :param vectorizer_min_df: min document frequency for text vectorizer
        :param vectorizer_max_df: max document frequency for text vectorizer
        :param n_estimators: the number of estimators (trees) to use
        :param use_sensor_code: use the supplied sensor code for classification
        :return: None
        """
        # Minimum/maximum document frequency for vectorizer
        # The default values seem to be close to optimal settings after some tuning
        self.vectorizer_min_df = vectorizer_min_df
        self.vectorizer_max_df = vectorizer_max_df

    def classify(self, text) -> List[ClassifierResult]:
        """
        Classify a record consisting of text and sensor codes
        :return The detected class as ClassifierResult
        """
        # print(sensor_codes)
        data_to_classify = self._create_data_table_for_doc(text)
        # print("\nData table: \n{0}\n".format(data_to_classify))
        matrix_counts = self.fitted_count_vectorizer.transform(data_to_classify.text)

        matrix_tf = self.fitted_tfidf_transformer.transform(matrix_counts)
        matrix_tf = matrix_tf.toarray()
        predicted = self.sklearn_classifier.predict(matrix_tf)
        predicted_class = predicted[0]

        result = ClassifierResult(predicted_class, -1, "")
        return [result]

    def train(self, training_data, class_label):
        """
        Train the algorithm with the data from the knowledge graph
        """
        self.training_data = training_data
        data_train = self._create_data_table_from_training_file(training_data, class_label)
        data_train = data_train.fillna(0)

        # We use str.split because the data has been pre-tokenised by us
        self.fitted_count_vectorizer = CountVectorizer(analyzer=str.split, min_df=self.vectorizer_min_df, max_df = self.vectorizer_max_df)
        matrix_train_counts = self.fitted_count_vectorizer.fit_transform(data_train.text)
        self.fitted_tfidf_transformer = TfidfTransformer(use_idf=True).fit(matrix_train_counts)
        matrix_train_tf = self.fitted_tfidf_transformer.transform(matrix_train_counts)
        matrix_train_tf = matrix_train_tf.toarray()
        self.sklearn_classifier.fit(matrix_train_tf, data_train.label)
        self._store_models()

    # ******************
    # Model management
    # *****************
    def _models(self):
        """
        Return a list of all variables which hold a part of the trained model
               + the filename to store the model
        """
        var2model = [
            ("fitted_count_vectorizer", "count_vectorizer.pkl"),
            ("sklearn_classifier", "classifier.pkl"),
            ("fitted_tfidf_transformer", "tfidf_transformer.pkl")
            ]

        return var2model

    def _create_model_folder_name(self, id):
        """
        Synthesize the name of the model folder for a given model id
        """
        return os.path.join(self.model_folder_path, "bc_model_{0:09d}".format(id))

    def _get_model_folder(self, new = False):
        """
        Return the model folder
            if new is False, return the model folder with the highest id
            if new is true, return the a new empty model folder (already created)
        """
        model_dirs = glob.glob(os.path.join(self.model_folder_path, "bc_model*"))
        model_dirs.sort()
        if len(model_dirs) > 0:
            last = model_dirs[len(model_dirs)-1]
        else:
            last = None

        if new is False:
            return last

        if(last):
            number = re.sub(".*bc_model_0*", "", last)
        else:
            number = "0"
        new_folder = self._create_model_folder_name(int(number) + 1)
        os.mkdir(new_folder)
        return new_folder

    def _store_models(self):
        """
        Store all object fitted to the training data
        """
        # Create a new model folder
        model_folder = self._get_model_folder(new = True)
        # Store all model components
        for (variable_name, file_name) in self._models():
            file_path = os.path.join(model_folder, file_name)
            model = getattr(self, variable_name)
            joblib.dump(model, file_path)

    def _load_models(self):
        # Get the newest model folder
        model_folder = self._get_model_folder(new = False)
        if not model_folder:
            return
        # Load all models by loading them to the variables
        for (variable_name, file_name) in self._models():
            file_path = os.path.join(model_folder, file_name)
            # print("Loading model {0}".format(file_name))
            setattr(self, variable_name, joblib.load(file_path))

    # ********************************
    # Creation  of data to classify
    # ********************************
    def _create_data_point(self, text):
        """
        Create a data point with the text and the given sensor codes
        If extra_features is not none, the ones not in the list will be added with value 1
        """
        datapoint = {}
        datapoint["text"] = self._tokenize(text)
        return datapoint

    def _create_data_table_for_doc(self, text):
        """
        Create a data table for a single PdfXml doc
        """
        datapoint = self._create_data_point(text)
        return self._create_data_table([datapoint])

    def _create_data_table_from_training_file(self, training_file, class_label):
        """
        Create a data table based on ADS knowledge graph (for training)
        """ 
        # Store failur
        with open(training_file, encoding = 'utf-8') as training_fp:
            datapoints = []
            for line in training_fp:
                record = json.loads(line)
                datapoint = self._create_data_point(record["text"])
                datapoint["label"] = record[class_label]
                datapoints.append(datapoint)

        return self._create_data_table(datapoints)

    def _create_data_table(self, datapoints):
        # shuffle for good measure
        datapoints = shuffle(datapoints)
        data_table = pandas.DataFrame(datapoints)
        return data_table

    def _tokenize(self, text):
        """
        Tokenizer. Hardcoded for Chinese. Change here for other languages
        """  
        tokenize = self._create_tokenizer()
        return " ".join(tokenize(text))

    def _create_tokenizer(self, lang="generic"):
        """
        Create and return a tokenisation function of type String->List[String].
        :param lang: an ISO 639-1 language code
        :return A function of type String->List[String]
        """
        tok_func = CountVectorizer().build_tokenizer()

        return tok_func


def main():
    """
    Just intended for testing
    """
    parser = argparse.ArgumentParser(description='Run various classifier from scikit learn')
    parser.add_argument('--training',
                    default = r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_training_data.json",
                    help='Data for training')
    parser.add_argument('--test',
                    default = r"D:\ProjectData\Uni\ltrs\data\classifier\classifier_test_data.json",
                    help='Data for test')

    args = parser.parse_args()

    label = "author"
    classifier = SklearnClassifier("RandomForestClassifier")
    print("INFO: Training")
    classifier.train(args.training, label)
    print("INFO: Training completed")

    with open(args.test, "r", encoding="utf-8") as test:
        good = 0
        bad = 0
        for line in test:
            json_data = json.loads(line)
            res = classifier.classify(json_data["text"])
            class_name = "none"
            if len(res) > 0:
                class_name = res[0].class_name
            if class_name == json_data[label]:
                #print("OK: {0}".format(class_name))
                good += 1
            else:
                print("BAD: {0} found {1} : {2}".format(json_data[label], class_name, json_data["text"]))
                bad += 1

        print("Good: {0}, Bad {1}".format(good, bad))


if __name__ == '__main__':
    main()


