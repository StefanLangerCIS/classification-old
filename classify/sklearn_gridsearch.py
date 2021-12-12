""" Grid search implementation
"""

# Sklearn: Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Sklearn: Other utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn_classifiers import create_data_table_from_training_file

from typing import Dict


class SklearnGridSearch:
    """
    Classify with sklearn classifiers
    """
    supported_classifiers = ["RandomForestClassifier"]

    def __init__(self, classifier_type :str, parameters: Dict, verbose=False):
        """
        Initialize the classifier
        :param classifier_type: The name of the classifiers
        :param grid: Parameter settings to check
        :param verbose: verbosity true or false
        """
        self.verbose = verbose

        if classifier_type == "RandomForestClassifier":
            self.sklearn_classifier = RandomForestClassifier()
            self.parameters = parameters

        else:
            raise Exception("Unsupported classifier type {0}. Use one of {1}".format(classifier_type, self.supported_classifiers))

        self.classifier_type = classifier_type
        self.training_data = None
        self.count_vectorizer = None
        self.tfidf_transformer = None

    def grid_search(self, training_data: str, text_label: str, class_label: str) -> Dict:
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

        data_train = create_data_table_from_training_file(training_data, text_label, class_label, 10000)
        print("INFO: grid evaluation with {0} data points".format(len(data_train)))
        data_train = data_train.fillna(0)
        self.count_vectorizer = CountVectorizer(min_df=10, max_df=0.8, ngram_range=(1, 1))
        matrix_train_counts = self.count_vectorizer.fit_transform(data_train.text)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(matrix_train_counts)
        matrix_train_tf = self.tfidf_transformer.transform(matrix_train_counts)
        matrix_train_tf = matrix_train_tf.toarray()

        grid_search = GridSearchCV(self.sklearn_classifier, self.parameters, n_jobs=10)
        grid_search.fit(matrix_train_tf, data_train.label)
        print(grid_search.best_params_)
        return grid_search.best_params_



