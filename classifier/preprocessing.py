import cv2
import re
import os
import pandas as pd
import numpy as np
import pandas as pd


def openImageAsArray( path ):

    return cv2.imread(path).flatten()


def listFilePaths( path ):
    ans = []
    files = os.listdir( path )
    for fileName in files:
        ans.append( path+"/"+fileName )

    return ans


def processFolder( pathToFolder ):
    answer = []
    files = listFilePaths(pathToFolder)

    for file in files:
        image = openImageAsArray(file)
        answer.append(image)

    return answer


def convertImagesToDataFrame( path ):
    classes = os.listdir( path )
    data = pd.DataFrame()
    entries = []
    labels = []

    for classe in classes:
        listOfImages = processFolder(path+"/"+classe)
        entries.extend( listOfImages )
        labels.extend( [classe]*len(listOfImages) )

    data = pd.DataFrame(entries)
    data["labels"] = labels

    return data


if __name__ == "__main__":
    path = "../database/FingerCamera"
    data = convertImagesToDataFrame(path)
    data.to_csv("data.csv")