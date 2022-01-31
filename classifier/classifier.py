import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':

    print("loading the data")

    data = pd.read_csv( "data.csv" )

    print("split x and y")

    y = data['labels']
    data.drop('labels', axis=1, inplace = True)

    print("defining the model")

    model = MLPClassifier( hidden_layer_sizes=(1,50) )

    print("spliting the data")

    trainx, testx, trainy, testy = train_test_split(data,y,train_size=0.7)

    print("training")

    model.fit( trainx, trainy )

    print("predictind ... ")

    predictions = model.predict( testx )
    #TODO: implementar funcao para medir a acauracia
    #mse = mean_squared_error(testy, predictions)
    #print("mean square error =", mse)