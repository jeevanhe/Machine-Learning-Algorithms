import sys
import pandas as pd


def cleanHouseData(housedata) :
    '''
    Removes rows of missing values/NAN,
    performs standardization on each value
    and returns preprocessed data
    '''

    # drop missing value row(s)
    housedata.dropna(inplace = True)
    # binary variable separation
    bindata = housedata[['CHAS']]
    # predicted variable separation
    outdata = housedata[['MEDV']]
    # drop above variable
    housedata.drop(['CHAS', 'MEDV'], axis=1, inplace=True)
    # standardize the numeric data
    stddata = (housedata - housedata.mean())/housedata.std(ddof=1)
    # merge all variables and return
    return stddata.join(bindata.join(outdata))


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
    houseData = pd.read_csv(inputPath, sep=r"\s+",
                            names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                   'B', 'LSTAT', 'MEDV'])
    # clean dataset
    modHouseData = cleanHouseData(houseData)
    # save cleaned dataset
    saveFile(modHouseData, outputPath, "HouseData")
