import sys
import pandas as pd


def cleanirisData(irisdata) :
    '''
    Removes rows of missing values/NAN,
    categorical variables/class label mapping/encoding,
    performs standardization on each value
    and returns preprocessed data
    '''

    # drop missing value row(s)
    irisdata.dropna(inplace = True)

    # class label encoding
    class_mapping = {'Iris-setosa': 1,
                    'Iris-versicolor': 2,
                    'Iris-virginica': 3}

    irisdata['CLASS'] = irisdata['CLASS'].map(class_mapping)

    # predicted class separation
    outdata = irisdata[['CLASS']]
    # drop above variable
    irisdata.drop(['CLASS'], axis=1, inplace=True)
    # standardize the numeric data
    stddata = (irisdata - irisdata.mean())/irisdata.std(ddof=1)
    # merge all variables and return
    return stddata.join(outdata)


def saveFile(savedata, savepath, dataset):
    '''
    Saves the data into given output path
    '''
    path = savepath + dataset + "PreProc.data"
    savedata.to_csv(path, index=False)


if __name__ == "__main__":

    # Fetch the system variables "housing.data"
    inputPath = sys.argv[1]
    # only directory location Ex: C:\abc\xyz\
    outputPath = sys.argv[2]

    # load dataset
    irisData = pd.read_csv(inputPath, sep=",",
                            names=['SEPALLENGTH', 'SEPALWIDTH', 'PETALLENGTH', 'PETALWIDTH','CLASS'])
    # clean dataset
    modirisData = cleanirisData(irisData)
    # save cleaned dataset
    saveFile(modirisData, outputPath, "irisData")
