import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np


def getUniqueValues( array ):

    return list(set(array))


def formatConfusionMatrix( predictions ):
    labels = getUniqueValues( predictions['expected'].values )
    answer = pd.DataFrame( confusion_matrix(predictions['expected'].values, predictions['predictions'].values, labels = labels), columns = labels, index = labels )

    return answer


def evaluateAccuracyPerClass( predictions ):
    classes = getUniqueValues(predictions['expected'])
    number_of_classes = len(classes)
    answer = pd.Series( np.zeros(number_of_classes, dtype=float) )
    answer.index = classes

    for className in classes:
        acc = accuracy_score( predictions['expected'][ predictions['expected']==className ], predictions['predictions'][ predictions['expected']==className ])
        print( "current class is", className )
        print("accuracy = ", acc)
        answer[className]=acc

    return answer


def createReports( predictions ):
    data = evaluateAccuracyPerClass( predictions )
    data.to_csv("class_accuracy.csv")
    data = formatConfusionMatrix( predictions )
    data.to_csv("confusion_matrix.csv")


def runModel():
    data = pd.read_csv( "data.csv" )
    ordinal_encoder = OrdinalEncoder()
    data['labels'] = ordinal_encoder.fit_transform(data['labels'].values.reshape(-1, 1))
    y = data['labels']
    data.drop('labels', axis=1, inplace = True)
    model = MLPClassifier( hidden_layer_sizes=(50,50,50), random_state = 7, max_iter = 1000 )
    #trainx, testx, trainy, testy
    trainx, testx, trainy, testy = train_test_split(data,y,train_size=0.7)
    model.fit( trainx, trainy )
    predictions = model.predict( testx )
    print(accuracy_score(predictions,testy) )
    print("shapes should match", predictions.shape,testy.shape)
    predictions = pd.DataFrame(ordinal_encoder.inverse_transform(predictions.reshape(-1, 1)))
    predictions["expected"] = ordinal_encoder.inverse_transform(testy.reshape(-1, 1))
    predictions.columns = ["predictions", "expected"]
    predictions.to_csv( "predictions.csv" )


if __name__ == '__main__':
    predictions = pd.read_csv("testing_data.csv")
    createReports(predictions)
    #runModel()