import abc
from abc import ABC
from typing import List


class ClassifierResult:
    def __init__(self, class_name, class_score = -1, meta_information = ""):
        """
        A classifier result
        class_name can be any string, in our context we expect a failure mode, taken from the knowledge graph
        class_score is any number. Optionally normalize to value between 0...1
        meta_information is any text you would like to add to the class
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
    def train(self, training_data, class_label):
        """
        Train using the training data
        Probably only the context --> failure_modes/symptoms part is needed for training
        """
        return

    @abc.abstractmethod
    def classify(self, text) -> List[ClassifierResult]:
        """
        Return an ordered list of ClassifierResults
        """
        return []



