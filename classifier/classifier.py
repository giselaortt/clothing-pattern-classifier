import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def getUniqueValues( array ):

    return list(set(array))


def formatConfusionMatrix( expected, predicted ):
    labels = getUniqueValues(expected)
    answer = pd.DataFrame( confusion_matrix(expected, predicted, labels), columns = labels, index = labels )

    return answer


def evaluateAccuracyPerClass( predictions, expected ):
    predictions = pd.DataFrame(predictions)
    expected = pd.DataFrame(expected)
    classes = getUniqueValues(expected)
    number_of_classes = len(classes)
    answer = pd.Series( np.zeros(number_of_classes, dtype=float) )
    answer.index = classes

    for className in classes:
        acc = accuracy_score( predictions[ expected==className ], expected[ expected==className ])
        print( "current class is", className )
        print("accuracy = ", acc)
        answer[className]=acc

    return answer


def createReports( expected, predicted ):
    data = evaluateAccuracyPerClass(expected, predicted)
    data.to_csv("class_accuracy.csv")
    data = formatConfusionMatrix( expected, predicted )
    data.to_csv("confusion_matrix.csv")


if __name__ == '__main__':

    print("loading the data")

    data = pd.read_csv( "data.csv" )

    print("preprocessing")

    ordinal_encoder = OrdinalEncoder()
    dataframe[columns] = ordinal_encoder.fit_transform(dataframe[columns])
    ordinal_encoder.fit_transform(classes)

    y = data['labels']
    data.drop('labels', axis=1, inplace = True)

    print("defining the model")

    model = MLPClassifier( hidden_layer_sizes=(1,50), random_state = 7 )

    print("spliting the data")

    trainx, testx, trainy, testy = train_test_split(data,y,train_size=0.7)

    print("training")

    model.fit( trainx, trainy )

    print("predictind ... ")

    predictions = model.predict( testx )

    print(accuracy_score(predictions,testy) )

    predictions = pd.DataFrame(ordinal_encoder.inverse_transform(predictions))
    predictions["expected"] = testy
    predictions.columns = ["predictions", "expected"]
    predictions.to_csv( "predictions.csv" )
