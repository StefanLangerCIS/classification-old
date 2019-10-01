import abc
from abc import ABC
from typing import List


class ClassifierResult:
    def __init__(self, class_name: str, class_score = -1, meta_information = ""):
        """
        :param class_name: can be any string, in our context we expect a failure mode, taken from the knowledge graph
        :param class_score: any number. Optimally normalized to value between 0...1
        :param meta_information: any text information you would like to add to the result
        """
        self.class_name = class_name
        self.class_score = class_score
        self.meta_information = meta_information

    def __repr__(self):
        return "{0} object. class_name: {1}, class_description: {2}, meta_information: {3}".format(self.__class__, self.class_name, self.class_score, self.meta_information)


class TextClassifier(ABC):
    """
    Abstract base class for ADS classifier + the definition of a classifier result
    """
    @abc.abstractmethod
    def train(self, training_data: str, text_label: str, class_label: str) -> None:
        """
        Train using the training data
        :param training_data: file with json data, one json record per line
        :param text_label: the json field which contains the text
        :param class_label:  the json field which contains the label
        :return: Nothing
        """
        return

    @abc.abstractmethod
    def classify(self, data, text_label: str) -> List[ClassifierResult]:
        """
        Classify a given text
        :param data: dictionary (parsed json) - the record to classify
        :param text_label: the dictionary/json field which ocntains the text
        :return: List of predicted classes (for most classifier, just one class)
        """
        """
        Return an ordered list of ClassifierResults
        """
        return []



