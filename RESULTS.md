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
			 
### Confusion Matrix (img)

![Confusion matrix](img/results_Perceptron_AuthorsWordBiGrams.jpg)

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
	 
### Confusion Matrix (img)

![Confusion matrix](img/results_Perceptron_AuthorsDefault_Settings.jpg)

## Classifier: Logistic Regression with word bi-grams
- Algorithm: Scikit-learn logistic regression
- Features: word bigrams in addition to single words
- Reported by: Shuzhou Yuan, Shanshan Bai

### Performance

- Seconds used for training: 80
- Seconds used for classification: 0.05
        
	
### Classification report
                            precision    recall  f1-score   support

               Franz Kafka       0.90      0.88      0.89      1017
        Friedrich Schiller       0.81      0.81      0.81       978
              Henrik Ibsen       1.00      0.98      0.99      3209
               James Joyce       0.96      0.88      0.92      2459
    Johann Wolfgang von Goethe   0.77      0.76      0.76       817
            Virginia Woolf       0.96      0.99      0.97      6872
             Wilhelm Busch       0.93      0.96      0.95      2232

                 micro avg       0.94      0.94      0.94     17584
                 macro avg       0.90      0.89      0.90     17584
              weighted avg       0.94      0.94      0.94     17584



### Confusion matrix
	[[ 900   18    0    0   28    0   71]
 	[  24  797    0    0  125    0   32]
 	[  10   10 3150    6   10    4   19]
 	[   3    0    4 2161    0  289    2]
 	[  24  140    0    0  618    1   34]
 	[   0    0    0   94    0 6778    0]
 	[  39   19    1    0   26    0 2147]]
			  
### Confusion Matrix (img)

![Confusion matrix](img/results_LogisticRegression_Authorbigramm.jpg)

## Classifier: Logistic Regression with single words
- Algorithm: Scikit-learn logistic regression
- Features: single words
- Reported by: Shuzhou Yuan, Shanshan Bai


### Performance: 

- Seconds used for training: 11
- Seconds used for classification: 0.01


### Classification report
                            precision    recall  f1-score   support

               Franz Kafka       0.92      0.86      0.89      1051
        Friedrich Schiller       0.80      0.81      0.80       978
              Henrik Ibsen       1.00      0.99      0.99      3188
               James Joyce       0.95      0.90      0.92      2500
    Johann Wolfgang von Goethe   0.76      0.76      0.76       803
            Virginia Woolf       0.97      0.98      0.97      6871
             Wilhelm Busch       0.93      0.96      0.95      2193

                 micro avg       0.94      0.94      0.94     17584
                 macro avg       0.90      0.89      0.90     17584
              weighted avg       0.94      0.94      0.94     17584


 
### Confusion matrix: 

	[[ 907   25    3    0   25    0   91]
 	[  20  790    0    0  143    0   25]
 	[   8   11 3143    5    4    3   14]
 	[   2    0    4 2253    2  238    1]
 	[  11  147    1    0  610    1   33]
 	[   0    0    0  125    0 6745    1]
 	[  43   19    1    0   16    0 2114]]
	 
### Confusion Matrix (img)

![Confusion matrix](img/results_LogisticRegression_Authorsingleword.jpg)

