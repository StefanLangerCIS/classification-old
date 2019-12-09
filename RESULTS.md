# Classification results for author classification
This page contains the classification results for the various algorithms, letter author classification

## Classifier: Perceptron with word bi-grams
- Algorithm: Scikit-learn perceptron
- Features: word bigrams in addition to single words
- Reported by: Stefan Langer

### Performance

- Seconds used for training: 180
- Seconds used for classification: 17

### Classification report
                                  precision    recall  f1-score   support

                     Franz Kafka       0.95      0.92      0.93       280
              Friedrich Schiller       0.80      0.77      0.79       266
                    Henrik Ibsen       0.97      0.99      0.98       897
                     James Joyce       0.97      0.92      0.95       682
      Johann Wolfgang von Goethe       0.70      0.80      0.75       228
                  Virginia Woolf       0.97      0.99      0.98      1901
                   Wilhelm Busch       0.99      0.93      0.96       627

                        accuracy                           0.95      4881
                       macro avg       0.91      0.90      0.90      4881
                    weighted avg       0.95      0.95      0.95      4881


### Confusion matrix
      [[ 258    3    7    0    7    0    5]
       [   1  205    7    0   52    0    1]
       [   1    1  892    1    1    0    1]
       [   0    0    0  629    0   53    0]
       [   3   39    3    0  182    0    1]
       [   0    0    1   19    0 1881    0]
       [   9    7   12    0   17    0  582]]

## Classifier: Perceptron with single words
- Algorithm: Scikit-learn perceptron
- Features: single words
- Reported by: Stefan Langer


### Performance: 

- Seconds used for training: 66
- Seconds used for classification: 18


### Classification report

                                precision    recall  f1-score   support

                   Franz Kafka       0.98      0.82      0.89       280
            Friedrich Schiller       0.66      0.91      0.76       266
                  Henrik Ibsen       0.99      0.99      0.99       897
                   James Joyce       0.93      0.93      0.93       682
    Johann Wolfgang von Goethe       0.78      0.64      0.71       228
                Virginia Woolf       0.98      0.98      0.98      1901
                 Wilhelm Busch       0.95      0.94      0.95       627

                      accuracy                           0.94      4881
                     macro avg       0.90      0.89      0.89      4881
                  weighted avg       0.95      0.94      0.94      4881


 
### Confusion matrix: 

	[[ 230   21    1    0    8    0   20]
	 [   0  241    0    0   22    0    3]
	 [   0    6  886    2    2    0    1]
	 [   0    0    1  637    0   44    0]
	 [   0   78    0    0  146    0    4]
	 [   0    0    1   44    0 1856    0]
	 [   4   21    2    0    8    0  592]]