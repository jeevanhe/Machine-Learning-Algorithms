import sys
import pandas as pd


def cleanadultData(adultdata) :
    '''
    Removes rows of missing values/NAN,
    categorical variables/class label mapping/encoding,
    performs standardization on each value
    and returns preprocessed data
    '''

    # class label encoding
    workclass_mapping = {'Private' : 1,
                         'Self-emp-not-inc' : 2,
                         'Self-emp-inc' : 3,
                         'Federal-gov' : 4,
                         'Local-gov' : 5,
                         'State-gov' : 6,
                         'Without-pay' : 7,
                         'Never-worked' : 8}

    education_mapping = {'Bachelors' : 1,
                         'Some-college' : 2,
                         '11th' : 3,
                         'HS-grad' : 4,
                         'Prof-school' : 5,
                         'Assoc-acdm' : 6,
                         'Assoc-voc' : 7,
                         '9th' : 8,
                         '7th-8th' : 9,
                         '12th' : 10,
                         'Masters' : 11,
                         '1st-4th' : 12,
                         '10th' : 13,
                         'Doctorate' : 14,
                         '5th-6th' : 15,
                         'Preschool' : 16}

    maritalstatus_mapping = {'Married-civ-spouse' :1,
                             'Divorced' : 2,
                             'Never-married' : 3,
                             'Separated' : 4,
                             'Widowed' : 5,
                             'Married-spouse-absent' : 6,
                             'Married-AF-spouse' : 7}

    occupation_mapping = {'Tech-support' : 1,
                          'Craft-repair' : 2,
                          'Other-service' : 3,
                          'Sales' : 4,
                          'Exec-managerial' : 5,
                          'Prof-specialty' : 6,
                          'Handlers-cleaners' : 7,
                          'Machine-op-inspct' : 8,
                          'Adm-clerical' : 9,
                          'Farming-fishing' : 10,
                          'Transport-moving' : 11,
                          'Priv-house-serv' : 12,
                          'Protective-serv' : 13,
                          'Armed-Forces' : 14}

    relationship_mapping = {'Wife' : 1,
                           'Own-child' : 2,
                           'Husband' : 3,
                           'Not-in-family' : 4,
                           'Other-relative' : 5,
                           'Unmarried' : 6}

    race_mapping = { 'White' : 1,
                     'Asian-Pac-Islander' : 2,
                     'Amer-Indian-Eskimo' : 3,
                     'Other' : 4,
                     'Black' : 5 }

    sex_mapping = {'Female' : 1,
                   'Male' : 2}

    nativecountry_mapping = {'United-States' : 1,
                             'Cambodia' : 2,
                             'England' : 3,
                             'Puerto-Rico' : 4,
                             'Canada' : 5,
                             'Germany' : 6,
                             'Outlying-US(Guam-USVI-etc)' : 7,
                             'India' : 8,
                             'Japan' : 9,
                             'Greece' : 10,
                             'South' : 11,
                             'China' : 12,
                             'Cuba' : 13,
                             'Iran' : 14,
                             'Honduras' : 15,
                             'Philippines' : 16,
                             'Italy' : 17,
                             'Poland' : 18,
                             'Jamaica' : 19,
                             'Vietnam' : 20,
                             'Mexico' : 21,
                             'Portugal' : 22,
                             'Ireland' : 23,
                             'France' : 24,
                             'Dominican-Republic' : 25,
                             'Laos' : 26,
                             'Ecuador' : 27,
                             'Taiwan' : 28,
                             'Haiti' : 29,
                             'Columbia' : 30,
                             'Hungary' : 31,
                             'Guatemala' : 32,
                             'Nicaragua' : 33,
                             'Scotland' : 34,
                             'Thailand' : 35,
                             'Yugoslavia' : 36,
                             'El-Salvador' : 37,
                             'Trinadad&Tobago' : 38,
                             'Peru' : 39,
                             'Hong' : 40,
                             'Holand-Netherlands' : 41}

    class_mapping = {'<=50K' : 1,
                     '>50K' : 2}

    # Encoding the categorical variables column-wise
    adultdata['workclass'] = adultdata['workclass'].map(workclass_mapping)
    adultdata['education'] = adultdata['education'].map(education_mapping)
    adultdata['marital-status'] = adultdata['marital-status'].map(maritalstatus_mapping)
    adultdata['occupation'] = adultdata['occupation'].map(occupation_mapping)
    adultdata['relationship'] = adultdata['relationship'].map(relationship_mapping)
    adultdata['race'] = adultdata['race'].map(race_mapping)
    adultdata['sex'] = adultdata['sex'].map(sex_mapping)
    adultdata['native-country'] = adultdata['native-country'].map(nativecountry_mapping)
    adultdata['class'] = adultdata['class'].map(class_mapping)

    # drop missing value row(s)
    adultdata.dropna(inplace = True)

    # categorial variable/class separation
    columnlist = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','class']
    outdata = adultdata[columnlist]

    # drop categorical varaibles from main dataframe
    adultdata.drop(columnlist, axis=1, inplace=True)

    # standardize the numeric data
    stddata = (adultdata - adultdata.mean())/adultdata.std(ddof=1)

    # merge all columns/variables and return
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
    adultData = pd.read_csv(inputPath, sep=",", skipinitialspace=True,
                            names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                   'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                   'hours-per-week', 'native-country', 'class'])
    # clean dataset
    modadultData = cleanadultData(adultData)
    # save cleaned dataset
    saveFile(modadultData, outputPath, "adultData")
