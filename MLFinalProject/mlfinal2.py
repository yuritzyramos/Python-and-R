import argparse
import numpy as np
import pandas as pd
import sklearn
import cpi
from difflib import SequenceMatcher

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor  # SGD Regression - Sam
from xgboost import XGBRegressor
from xgboost import plot_importance # plot most important features

from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from sklearn import metrics  # for calculating accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

#Program written by Sabrina Martinez, Samantha Lin, & Yuritzy Ramos

# ==================================================used functions========================================#

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def stringcomp(name, platform, genre, publisher, developer, rating, videoGameDataset):
    for i in range(videoGameDataset.shape[0]):
        videoGameDataset.at[i, 'Name'] = similar(name, videoGameDataset.iloc[i]['Name'])
        videoGameDataset.at[i, 'Platform'] = similar(platform, videoGameDataset.iloc[i]['Platform'])
        videoGameDataset.at[i, 'Genre'] = similar(genre, videoGameDataset.iloc[i]['Genre'])
        videoGameDataset.at[i, 'Publisher'] = similar(publisher, videoGameDataset.iloc[i]['Publisher'])
        videoGameDataset.at[i, 'Developer'] = similar(developer, videoGameDataset.iloc[i]['Developer'])
        videoGameDataset.at[i, 'Rating'] = similar(rating, videoGameDataset.iloc[i]['Rating'])

    return videoGameDataset


def estimateRevenue(videoGameDataset, averageCostOfGames):
    # videoGameDataSet= pandas dataframe of games
    # averageCostOfGames= self explanatory, dictionary, platform: price

    # return estimated revenues adjusted for current inflation
    yHat = []
    # ====#
    a = videoGameDataset.copy()
    a['Year_of_Release'] = pd.to_datetime(a['Year_of_Release'], format='%Y')

    for i in range(a.shape[0]):
        totalsales = a.iloc[i]['Global_Sales']
        price = averageCostOfGames[a.iloc[i]['Platform']]
        year = a.iloc[i]['Year_of_Release'].year

        revenue = totalsales * price

        adjustedRevenue = cpi.inflate(revenue, year, to=2021)

        yHat.append(adjustedRevenue)

    yHat = np.array(yHat)

    yHat = pd.DataFrame(yHat)

    return yHat


def estimateIndustryCategory(videoGameDataset, developerStudios, publishingStudios):
    # videoGameDataSet= pandas dataframe of games
    # developerStudios= self explanatory, list of names of developers
    # publishingStudios= self explanatory, list of names of publishers

    # return wether a game is indie or AAA
    yHat = []

    # ====#

    for i in range(videoGameDataset.shape[0]):

        publisher = videoGameDataset.iloc[i]['Publisher']
        developer = videoGameDataset.iloc[i]['Developer']
        globalSales = videoGameDataset.iloc[i]['Global_Sales']

        if publisher in publishingStudios or developer in developerStudios or globalSales > 2:
            yHat.append(1)
        else:
            yHat.append(0)

    yHat = np.array(yHat)

    yHat = pd.DataFrame(yHat)

    return yHat


def estimateSuccess(videoGameDataset, yIndieAAAlabel):
    # videoGameDataSet= pandas dataframe of games
    # yIndieAAAlabel= labels of wether a game is indie or not

    # return wether a game is indie or AAA
    yHat = []

    # ====#

    for i in range(videoGameDataset.shape[0]):

        globalSales = videoGameDataset.iloc[i]['Global_Sales']

        indieoraaa = yIndieAAAlabel.iloc[i][0]

        if indieoraaa:  # aaa
            if globalSales >= 2:
                yHat.append(1)
            else:
                yHat.append(0)

        else:  # indie
            if globalSales >= 0.02:
                yHat.append(1)
            else:
                yHat.append(0)

    yHat = np.array(yHat)

    yHat = pd.DataFrame(yHat)

    return yHat


# ===================================================branch functions=====================================#

def regressionalOld(videoGameDataset, yRevenue, fakeVideoGameDF):
    # Old regressional attempt 1, using all features except critic/user score & count
    # This got very poor MSE, so this isn't used
    # Samantha's part
    # print("DATA:", videoGameDataset)
    copyDF = videoGameDataset.copy()
    copyDF2 = copyDF.drop(['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], axis=1, inplace=False)
    # copyDF2 = copyDF.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], axis=1, inplace=False)  # drop sales, not recommended

    # extract year
    # copyDF['Year_of_Release'] = pd.DatetimeIndex(copyDF['Year_of_Release']).year

    # convert pandas to numpy
    videoGameDatasetNP = copyDF2.to_numpy()
    copyY = yRevenue.to_numpy()

    # make sure to scale again - MinMax
    scaler = MinMaxScaler()
    scaledX = scaler.fit_transform(videoGameDatasetNP)

    # split train/test by 70/30, with fixed random state
    xTrain, xTest, yTrain, yTest = train_test_split(scaledX, copyY, train_size=0.7, random_state=42)

    # SGD Linear Regression Model, epoch = 4000, rest are default, do not use 'optimal'
    model = SGDRegressor(max_iter=4000, random_state=42)
    # model = LinearRegression()
    model.fit(xTrain, yTrain.ravel())

    # evaluate test split
    yPred = model.predict(xTest)

    # Print statements for debugging
    # print(xTrain)
    # print(yTrain)
    # print("TEST:", xTest)
    # print("Prediction:", yPred)
    # print("Actual:", yTest)
    # for i in range(len(yPred)):
    #     print("Prediction:", yPred[i], "Actual:", yTest[i][0])

    # MSE and RSE
    mse = metrics.mean_squared_error(yTest, yPred)
    print("SGD Regression MSE: ", mse)
    print("SGD Regression RMSE: ", mse ** (1 / 2.0))

    # # plot yTest and yPred revenue
    x_ax = range(len(yTest))
    plt.plot(x_ax, yTest, linewidth=1, label="actual")
    plt.plot(x_ax, yPred, linewidth=1.1, label="predicted")
    plt.title("Actual and Predicted Revenue SGD")
    plt.xlabel('X-axis')
    plt.ylabel('Revenue')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()

    # Regular Linear Regression Model
    model2 = LinearRegression()
    model2.fit(xTrain, yTrain.ravel())

    # evaluate test split
    yPred2 = model2.predict(xTest)

    # Print statements for debugging
    # print(xTrain)
    # print(yTrain)
    # print("TEST:", xTest)
    # print("Prediction:", yPred2)
    # print("Actual:", yTest)
    # for i in range(len(yPred2)):
    #     print("Prediction:", yPred2[i], "Actual:", yTest[i][0])

    # MSE and RSE
    mse2 = metrics.mean_squared_error(yTest, yPred2)
    print("Linear Regression MSE: ", mse2)
    print("Linear Regression RMSE: ", mse2 ** (1 / 2.0))

    # # plot yTest and yPred revenue
    x_ax = range(len(yTest))
    plt.plot(x_ax, yTest, linewidth=1, label="actual")
    plt.plot(x_ax, yPred2, linewidth=1.1, label="predicted")
    plt.title("Actual and Predicted Revenue Linear Regression")
    plt.xlabel('X-axis')
    plt.ylabel('Revenue')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()

    # fake video game test
    # copyFakePD = fakeVideoGameDF.copy()
    # copyFakePD.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], axis=1, inplace=True)  # drop sales
    # copyFakePD.drop(['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count'], axis=1, inplace=True)  # this will get feature mismatch error
    # copyFake = copyFakePD.to_numpy()
    #
    # yFake1 = model.predict(copyFake)
    # yFake2 = model2.predict(copyFake)
    # print("Fake Game (SGD) Predicted Revenue:", yFake1)
    # print("Fake Game (Linear) Predicted Revenue:", yFake2)
    return model

def regressional(yRevenue, originalDataset,fakeVideoGameDF, videoGameDataset):
    # insert regressional here
    # Samantha's part

    print("start regressional")
    copy= originalDataset.copy()
    copyFake= fakeVideoGameDF.copy()
    copy['Label']=  yRevenue.iloc[:, 0]
    data= copy.dropna()

    data.reset_index
    # only uses sales features
    x = data[["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]]
    y = pd.DataFrame(data['Label'])

    print(x)
    print(y)

    X_train, X_test, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train) # normalize - fit and transform training
    X_test = scaler.transform(X_test) # transform testing

    model = XGBRegressor(n_estimators=700, learning_rate=0.08, max_depth=5)  # best: lr = 0.08, max_depth = 5, nestimators = 700
    # model = XGBRegressor(n_estimators = 200, learning_rate= 0.08)
    model.fit(X_train, ytrain)

    # Hyperparameter tuning GridSearchCV - xgboost
    # param_gridXGB = {"max_depth": [4, 5, 6, 7],
    #               "n_estimators": [200, 300, 400, 500, 600, 700],
    #               "learning_rate": [0.1, 0.08, 0.01, 0.015, 0.001, 0.0001]}
    #
    # search = GridSearchCV(estimator=XGBRegressor(), param_grid=param_gridXGB, scoring='neg_mean_squared_error', verbose=1)
    # search.fit(X_train, ytrain)
    # print("XGB Best hyperparameters: ", search.best_params_)
    # print('XGB Best MSE (neg):', search.best_score_)

    # plot most important features XGB
    plt.style.use('fivethirtyeight')
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(12, 6))
    model.get_booster().feature_names = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    # xgboost.plot_importance(model.get_booster())
    plot_importance(model.get_booster(), max_num_features=None, ax=ax)
    plt.show()

    y_pred = model.predict(X_test)
    # print pred vs actual
    # for i in range(len(y_pred)):
    #     print("Predictions:", y_pred[i])

    # MSE and RSE
    mse = metrics.mean_squared_error(ytest, y_pred)
    print("XGB Regression MSE: ", mse)
    print("XGB Regression RMSE: ", mse ** (1 / 2.0))

    print("===")
    ytrain = ytrain.to_numpy() # convert to numpy to flatten
    model2 = LinearRegression()
    model2.fit(X_train, ytrain.ravel())
    y_pred2 = model2.predict(X_test)
    # MSE and RSE
    mse2 = metrics.mean_squared_error(ytest, y_pred2)
    print("Linear Regression MSE: ", mse2)
    print("Linear Regression RMSE: ", mse2 ** (1 / 2.0))

    print("===")

    model3 = SGDRegressor(alpha=0.00001, learning_rate='optimal', max_iter=2000, penalty='l2') # optimal params
    # model3 = SGDRegressor(alpha=0.00001, learning_rate='optimal', max_iter=1000, penalty='l1') # also optimal params
    model3.fit(X_train, ytrain.ravel())
    y_pred3 = model3.predict(X_test)
    # MSE and RSE
    mse3 = metrics.mean_squared_error(ytest, y_pred3)
    print("SGD Regression MSE: ", mse3)
    print("SGD Regression RMSE: ", mse3 ** (1 / 2.0))

    # hyperparameter tuning GridSearchCV - SGD
    # param_grid = {'alpha': [0.1, 0.001, 0.0001, 0.00001], 'max_iter': [1000, 2000, 5000, 7000],
    #               'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'], 'penalty': ['l2', 'l1', 'None']}
    # grid = GridSearchCV(estimator=SGDRegressor(), param_grid=param_grid, scoring='neg_mean_squared_error', verbose=1)
    # grid_result = grid.fit(X_train, ytrain.ravel())
    # print('SGD Best MSE (negative):', grid_result.best_score_)
    # print('SGD Best Hyperparameters:', grid_result.best_params_)

    return None


def categorical(originalDataset, fakeVideoGameDF, videoGameDataset, yIndieAAAlabel, ySuccessLabels):
    print("")
    print("Starting categorical!")

    # SABRINAS PART

    # sabrinasPrediction= indieOrAA(originalDataset,fakeVideoGameDF, videoGameDataset, yIndieAAAlabel)
    # print("\n SABRINA has predicted: "+ str(sabrinasPrediction))

    sabrinasPrediction = 0

    # COPYING AND SPLITTING THE ORIGINAL DATAFRAME

    copy = originalDataset.copy()

    copy['Label'] = yIndieAAAlabel.iloc[:, 0]
    copy['LabelSucc'] = ySuccessLabels.iloc[:, 0]

    originalIndies = copy[copy['Label'] == 0].copy()
    originalAaas = copy[copy['Label'] == 1].copy()

    # GETTING THE LABELS
    yIndies = pd.DataFrame(originalIndies.loc[:, "LabelSucc"])
    yAAA = pd.DataFrame(originalAaas.loc[:, "LabelSucc"])

    originalIndies.drop(['Label', 'LabelSucc'], axis=1, inplace=True)
    originalAaas.drop(['Label', 'LabelSucc'], axis=1, inplace=True)

    # COPYING AND SPLITTING THE STRING COMPARISON DATAFRAME

    copy = videoGameDataset.copy()
    copy['Label'] = yIndieAAAlabel.iloc[:, 0]
    copy['LabelSucc'] = ySuccessLabels.iloc[:, 0]

    fakeGameIndies = copy[copy['Label'] == 0].copy()
    fakeGameAaas = copy[copy['Label'] == 1].copy()

    fakeGameIndies.drop(['Label', 'LabelSucc'], axis=1, inplace=True)
    fakeGameAaas.drop(['Label', 'LabelSucc'], axis=1, inplace=True)

    if sabrinasPrediction == 0:  # indie
        stringCompUsed = fakeGameIndies.copy()
        stringCompY = yIndies.copy()
    elif sabrinasPrediction == 1:
        stringCompUsed = fakeGameAaas.copy()
        stringCompY = yAAA.copy()

    successful(fakeVideoGameDF, stringCompUsed, stringCompY, originalIndies, originalAaas, yIndies, yAAA)

    return None


def indieOrAA(originalDataset, fakeVideoGameDF, videoGameDataset, yIndieAAAlabel):
    # returns a single int= 0 for indie or 1 for triple AAA for the fake game

    resultsCount = {0: 0, 1: 0}

    # ============+++++++++++++++===============#
    # ============ENCODE & SPLIT ORIGINAL FOR TEST AND TRAIN==============#
    copyorigi = originalDataset.copy()

    ohe = OneHotEncoder()
    encodedcopyorigi = pd.DataFrame(ohe.fit_transform(copyorigi).toarray())

    # =============test and train split encoded dataframe===============#

    print(type(encodedcopyorigi))
    encodedcopyorigi['Label'] = yIndieAAAlabel.iloc[:,
                                0]  # add y column to dataframe so that randomization is made easier

    train, test = train_test_split(encodedcopyorigi, test_size=0.3)

    yTrain = pd.DataFrame(train.loc[:, "Label"])
    yTest = pd.DataFrame(test.loc[:, "Label"])

    encodedcopyorigi.drop('Label', axis=1, inplace=True)
    train.drop('Label', axis=1, inplace=True)
    test.drop('Label', axis=1, inplace=True)

    # ============PREPARE COPIES FOR FAKE GAME PREDICTION==============#

    copyDF = videoGameDataset.copy()
    copyFake = fakeVideoGameDF.copy()

    copyDF.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count',
                 'User_Score', 'User_Count'], axis=1, inplace=True)
    copyFake.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count',
                   'User_Score', 'User_Count'], axis=1, inplace=True)

    # =============SCALE======================#
    scaler = StandardScaler()
    scaler.fit(copyDF)

    # scopyX= scaled copy of dataframe X
    scopyDF = scaler.transform(copyDF)
    scopyDF = pd.DataFrame(copyDF, columns=copyDF.columns)

    scopyFake = scaler.transform(copyFake)
    scopyFake = pd.DataFrame(copyFake, columns=copyDF.columns)

    # ================================== KNN==================================#
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model using the training sets
    model.fit(scopyDF, np.ravel(yIndieAAAlabel))

    # Predict Output
    predicted = model.predict(scopyFake)  # 0:Indie, 1: AAA

    print("\n================KNN====================\n")
    print("\n\n IS THE FAKE GAME INDIE OR AAA? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME IS A AAA GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME IS AN INDIE GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    model.fit(train, np.ravel(yTrain))

    predicted = model.predict(test)

    print("\n\n WHAT DID KNN PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTest, predicted)
    prec = precision_score(yTest, predicted)
    rec = recall_score(yTest, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    # ============================DT==============================#
    model = DecisionTreeClassifier()

    # Train the model using the training sets
    model.fit(scopyDF, np.ravel(yIndieAAAlabel))

    # Predict Output
    predicted = model.predict(scopyFake)  # 0:Indie, 1: AAA

    print("\n================DECISION TREE====================\n")
    print("IS THE FAKE GAME INDIE OR AAA? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME IS A AAA GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME IS AN INDIE GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    model.fit(train, np.ravel(yTrain))

    predicted = model.predict(test)

    print("\n\n WHAT DID DT PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTest, predicted)
    prec = precision_score(yTest, predicted)
    rec = recall_score(yTest, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    # ============================RFC==============================#

    model = RandomForestClassifier(max_depth=30,
                                   max_features='auto',
                                   min_samples_leaf=4,
                                   min_samples_split=10,
                                   random_state=42)

    # Train the model using the training sets
    model.fit(scopyDF, np.ravel(yIndieAAAlabel))

    # Predict Output
    predicted = model.predict(scopyFake)  # 0:Indie, 1: AAA

    print("\n================RANDOM FOREST====================\n")
    print("IS THE FAKE GAME INDIE OR AAA? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME IS A AAA GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME IS AN INDIE GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    model.fit(train, np.ravel(yTrain))

    predicted = model.predict(test)

    print("\n\n WHAT DID RF PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTest, predicted)
    prec = precision_score(yTest, predicted)
    rec = recall_score(yTest, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    # ============================NB==============================#

    model = MultinomialNB()

    # Train the model using the training sets
    model.fit(scopyDF, np.ravel(yIndieAAAlabel))

    # Predict Output
    predicted = model.predict(scopyFake)  # 0:Indie, 1: AAA

    print("\n================NAIVE BAYES====================\n")
    print("IS THE FAKE GAME INDIE OR AAA? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME IS A AAA GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME IS AN INDIE GAME!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    model.fit(train, np.ravel(yTrain))

    predicted = model.predict(test)

    print("\n\n WHAT DID NB PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTest, predicted)
    prec = precision_score(yTest, predicted)
    rec = recall_score(yTest, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    if resultsCount[1] > resultsCount[0]:
        return 1
    else:
        return 0


def successful(fakeVideoGameDF, stringCompUsed, stringCompY, originalIndies, originalAaas, yIndies, yAAA):
    # Yuritzy's part
    # returns a single int= 0 for indie or 1 for triple AAA for the fake game

    resultsCount = {0: 0, 1: 0}

    # ============+++++++++++++++===============#

    # ============ENCODE & SPLIT ORIGINAL FOR TEST AND TRAIN==============#
    copyorigiIndies = originalIndies.copy()
    copyorigiAaas = originalAaas.copy()

    ohe = OneHotEncoder()
    encodedcopyorigiIndies = pd.DataFrame(ohe.fit_transform(copyorigiIndies).toarray())
    encodedcopyorigiAaas = pd.DataFrame(ohe.fit_transform(copyorigiAaas).toarray())

    # =============test and train split encoded dataframe===============#
    yIndies = yIndies.reset_index(drop=True)
    yAAA = yAAA.reset_index(drop=True)

    encodedcopyorigiIndies['Label'] = yIndies.iloc[:,
                                      0]  # add y column to dataframe so that randomization is made easier
    encodedcopyorigiAaas['Label'] = yAAA.iloc[:, 0]  # add y column to dataframe so that randomization is made easier

    trainIndies, testIndies = train_test_split(encodedcopyorigiIndies, test_size=0.3)
    trainAAA, testAAA = train_test_split(encodedcopyorigiAaas, test_size=0.3)

    yTrainIndies = pd.DataFrame(trainIndies.loc[:, "Label"])
    yTestIndies = pd.DataFrame(testIndies.loc[:, "Label"])

    yTrainAAA = pd.DataFrame(trainAAA.loc[:, "Label"])
    yTestAAA = pd.DataFrame(testAAA.loc[:, "Label"])

    encodedcopyorigiIndies.drop('Label', axis=1, inplace=True)
    trainIndies.drop('Label', axis=1, inplace=True)
    testIndies.drop('Label', axis=1, inplace=True)

    encodedcopyorigiAaas.drop('Label', axis=1, inplace=True)
    trainAAA.drop('Label', axis=1, inplace=True)
    testAAA.drop('Label', axis=1, inplace=True)

    # ============PREPARE COPIES FOR FAKE GAME PREDICTION==============#

    copyDF = stringCompUsed.copy()
    copyFake = fakeVideoGameDF.copy()

    copyDF.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count',
                 'User_Score', 'User_Count'], axis=1, inplace=True)
    copyFake.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count',
                   'User_Score', 'User_Count'], axis=1, inplace=True)

    # =============SCALE======================#
    scaler = StandardScaler()
    scaler.fit(copyDF)

    # scopyX= scaled copy of dataframe X
    scopyDF = scaler.transform(copyDF)
    scopyDF = pd.DataFrame(copyDF, columns=copyDF.columns)

    scopyFake = scaler.transform(copyFake)
    scopyFake = pd.DataFrame(copyFake, columns=copyDF.columns)

    # ================================== KNN==================================# DO THIS FOR EVERY MODEL

    model = KNeighborsClassifier(n_neighbors=5)  # CHANGE THIS

    # Train the model using the training sets
    model.fit(scopyDF, np.ravel(stringCompY))

    # Predict Output
    predicted = model.predict(scopyFake)  # 0:Bust, 1: Success

    print("\n================KNN====================\n")
    print("\n\n IS THE FAKE SUCCESSFUL OR A BUST? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME WILL BE A SUCCESS!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME WILL BE A BUST!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    print("==========INDIES ACCURACY=======")

    model.fit(trainIndies, np.ravel(yTrainIndies))

    predicted = model.predict(testIndies)

    print("\n\n WHAT DID KNN PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestIndies, predicted)
    prec = precision_score(yTestIndies, predicted)
    rec = recall_score(yTestIndies, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("==========AAAS ACCURACY=======")

    model.fit(trainAAA, np.ravel(yTrainAAA))

    predicted = model.predict(testAAA)

    print("\n\n WHAT DID KNN PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestAAA, predicted)
    prec = precision_score(yTestAAA, predicted)
    rec = recall_score(yTestAAA, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")


    #================================== DT==================================# DO THIS FOR EVERY MODEL
    dt = KNeighborsClassifier(n_neighbors=5)  # CHANGE THIS

    # Train the model using the training sets
    dt.fit(scopyDF, np.ravel(stringCompY))

    # Predict Output
    predicted = dt.predict(scopyFake)  # 0:Bust, 1: Success
    print("\n================DT====================\n")
    print("\n\n IS THE FAKE SUCCESSFUL OR A BUST? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME WILL BE A SUCCESS!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME WILL BE A BUST!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    print("==========INDIES ACCURACY=======")

    dt.fit(trainIndies, np.ravel(yTrainIndies))

    predicted = dt.predict(testIndies)

    print("\n\n WHAT DID DT PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestIndies, predicted)
    prec = precision_score(yTestIndies, predicted)
    rec = recall_score(yTestIndies, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("==========AAAS ACCURACY=======")
    dt.fit(trainAAA, np.ravel(yTrainAAA))

    predicted = dt.predict(testAAA)

    print("\n\n WHAT DID DT PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestAAA, predicted)
    prec = precision_score(yTestAAA, predicted)
    rec = recall_score(yTestAAA, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    # ================================== RF==================================# DO THIS FOR EVERY MODEL
    rf = RandomForestClassifier()  # CHANGE THIS

    # Train the model using the training sets
    rf.fit(scopyDF, np.ravel(stringCompY))

    # Predict Output
    predicted = rf.predict(scopyFake)  # 0:Bust, 1: Success
    print("\n================RF====================\n")
    print("\n\n IS THE FAKE SUCCESSFUL OR A BUST? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME WILL BE A SUCCESS!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME WILL BE A BUST!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    print("==========INDIES ACCURACY=======")

    rf.fit(trainIndies, np.ravel(yTrainIndies))

    predicted = rf.predict(testIndies)

    print("\n\n WHAT DID RF PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestIndies, predicted)
    prec = precision_score(yTestIndies, predicted)
    rec = recall_score(yTestIndies, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("==========AAAS ACCURACY=======")
    rf.fit(trainAAA, np.ravel(yTrainAAA))

    predicted = rf.predict(testAAA)

    print("\n\n WHAT DID RF PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestAAA, predicted)
    prec = precision_score(yTestAAA, predicted)
    rec = recall_score(yTestAAA, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    # ================================== NB ==================================#
    nb =MultinomialNB()

    # Train the model using the training sets
    nb.fit(scopyDF, np.ravel(stringCompY))

    # Predict Output
    predicted = nb.predict(scopyFake)  # 0:Bust, 1: Success
    print("\n================NB====================\n")
    print("\n\n IS THE FAKE SUCCESSFUL OR A BUST? \n")
    print("RESULTS: \n")
    print(predicted)

    if predicted[0]:
        print("THE FAKE GAME WILL BE A SUCCESS!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    else:
        print("THE FAKE GAME WILL BE A BUST!\n")
        resultsCount[predicted[0]] = resultsCount[predicted[0]] + 1

    print("==========INDIES ACCURACY=======")

    nb.fit(trainIndies, np.ravel(yTrainIndies))

    predicted = nb.predict(testIndies)

    print("\n\n WHAT DID NB PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestIndies, predicted)
    prec = precision_score(yTestIndies, predicted)
    rec = recall_score(yTestIndies, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("==========AAAS ACCURACY=======")
    nb.fit(trainAAA, np.ravel(yTrainAAA))

    predicted = nb.predict(testAAA)

    print("\n\n WHAT DID NB PREDICT ON THE TEST DATA? \n")
    print("RESULTS: \n")
    print(predicted)

    acc = accuracy_score(yTestAAA, predicted)
    prec = precision_score(yTestAAA, predicted)
    rec = recall_score(yTestAAA, predicted)

    print('ACCURACY: ')
    print(acc)

    print('PRECISION: ')
    print(prec)

    print('RECALL')
    print(rec)

    print("\n====================================\n")

    if resultsCount[1]>resultsCount[0]:
        return 1
    else:
        return 0


# ===================================================main functions=====================================#
def main():
    parser = argparse.ArgumentParser()

    # ==========MODEL ARGS==========#

    parser.add_argument("videoGameDataset",
                        default="VideoGameDataset.csv",
                        help="filename for dataset to be used in training and testing, must have same attributes as default",
                        nargs='?')
    parser.add_argument("model",
                        default="both",
                        help="what model do you want? categorical, regressional, or both", nargs='?')

    # ==========FAKE VIDEO GAME DETAILS ARGS==========#

    parser.add_argument("Name",
                        default="Undertale",
                        help="Name of fake game", nargs='?')
    parser.add_argument("Platform",
                        default="PC",
                        help="Console of fake game", nargs='?')
    parser.add_argument("Year",
                        default=2015,
                        help="Year of fake games release", nargs='?')
    parser.add_argument("Genre",
                        default="Role-Playing",
                        help="Genre of fake game", nargs='?')
    parser.add_argument("Publisher",
                        default="tobyfox",
                        help="Publisher of fake game", nargs='?')
    parser.add_argument("Developer",
                        default="tobyfox",
                        help="Publisher of fake game", nargs='?')
    parser.add_argument("Rating",
                        default="E10+",
                        help="ESBR of fake game", nargs='?')

    # ==========upon release, US currency==========#
    # estimated (IE no records found): WS, NG, PCFX

    averageCostOfGames = {"Wii": 50, "NES": 60, "GB": 40, "DS": 40,
                          "X360": 60, "PS3": 60, "PS2": 50, "SNES": 65,
                          "GBA": 40, "PS4": 60, "3DS": 40, "N64": 55,
                          "PS": 50, "XB": 50, "PC": 60, "2600": 30,
                          "PSP": 20, "XOne": 60, "WiiU": 50, "GC": 50,
                          "GEN": 65, "DC": 50, "PSV": 50, "SAT": 50,
                          "SCD": 50, "WS": 30, "NG": 90, "TG16": 50,
                          "3DO": 50, "GG": 30, "PCFX": 50}

    """inflation={ 1989:4.83, 1990:5.40, 1991:4.24, 1992:3.03, 1993:2.95,
                1994:2.61, 1995:2.81, 1996:2.93, 1997:2.34, 1998:1.55,
                1999:2.19, 2000:3.38, 2001:2.83, 2002:1.59, 2003:2.27,
                2004:2.68, 2005:3.39, 2006:3.23, 2007:2.85, 2008:3.84,
                2009:-0.36,2010:1.64, 2011:3.16, 2012:2.07, 2013:1.46,
                2014:1.62, 2015:0.12, 2016:1.26, 2017:2.13, 2018:2.44,
                2019:1.81, 2020:1.23, 2021:4.70, 2022:7.40}
    """

    # ==========top 50 dev studios (with added variations), according to IGN==========#

    developerStudios = ["Atari", "Bethesda Game Studios", "BioWare",
                        "Black Isle Studios", "Blizzard Entertainment",
                        "Brøderbund", "Bungie", "Capcom", "EA Canada",
                        "EA Digital Illusions CE", "Enix", "Epic Games",
                        "Game Freak", "HAL Laboratory", "Harmonix Music Systems",
                        "Id Software", "Infinity Ward", "Insomniac Games",
                        "Intelligent Systems", "Irrational Games", "Konami",
                        "Level-5", "Looking Glass Studios", "LucasArts",
                        "Maxis", "MicroProse", "Midway", "Namco",
                        "Naughty Dog", "Neversoft", "Nintendo EAD",
                        "Nintendo", "Origin Systems", "Polyphony Digital",
                        "PopCap Games", "Rare", "Rare Ltd.",
                        "Relic Entertainment", "Retro Studios",
                        "Rockstar North", "SCE Japan Studio", "SCE Santa Monica Studio",
                        "Sega AM2", "Sega", "Sierra Entertainment",
                        "SNK", "SONIC Team", "SquareSoft", "Thatgamecompany",
                        "Treasure", "Ubisoft Montreal", "Ubisoft",
                        "Valve", "Westwood Studios"]

    # ==========top publishing studios==========#

    publishingStudios = ["Tencent Games", "Sony Interactive Entertainment", "Sony",
                         "Sony Computer Entertainment", "Microsoft", "THQ",
                         "Activision Blizzard", "Activision", "Electronic Arts",
                         "EA", "Nintendo", "Bandai Namco", "Take-Two Interactive",
                         "Ubisoft", "Square Enix", "Konami", "Atari"
                                                             "Sega", "Capcom", "Warner Bros. Interactive Entertainment",
                         "Konami Digital Entertainment", "Namco Bandai Games"]

    # ==========each generation of consoles last only 5 years==========#

    generationLifeExpectancy = 5

    # ==========percental deprecation of games cost over the years due to new console generations,estimate==========#

    pricecuts = .5

    # ==========make yHats==========#

    yRevenue = []  # each cell is it's respective sample's (estimated revenue)
    yIndieAAAlabel = []  # indie=0, AAA=1
    ySuccessLabels = []  # not=0, success=1

    # ==========get args==========#
    args = parser.parse_args()

    videoGameDataset = pd.read_csv(args.videoGameDataset)

    videoGameDataset = videoGameDataset.dropna(
        subset=['Year_of_Release'])  # dropped ones with no year because i am too lazy
    videoGameDataset = videoGameDataset.dropna(subset=['Name'])

    videoGameDataset = videoGameDataset.reset_index(drop=True)

    videoGameDataset['Genre'] = videoGameDataset['Genre'].fillna('Unknown')
    videoGameDataset['Publisher'] = videoGameDataset['Publisher'].fillna('Unknown')
    videoGameDataset['Developer'] = videoGameDataset['Developer'].fillna('Unknown')
    videoGameDataset['Rating'] = videoGameDataset['Rating'].fillna('Unknown')

    originalDataset = videoGameDataset.copy()

    model = args.model
    name = args.Name
    platform = args.Platform
    year = args.Year
    genre = args.Genre
    publisher = args.Publisher
    developer = args.Developer
    rating = args.Rating

    # ==========fill yHats ==========#

    yRevenue = estimateRevenue(videoGameDataset, averageCostOfGames)  # in the millions
    yIndieAAAlabel = estimateIndustryCategory(videoGameDataset, developerStudios, publishingStudios)
    ySuccessLabels = estimateSuccess(videoGameDataset, yIndieAAAlabel)

    # ==========string similarity using difflib==========#

    videoGameDataset = stringcomp(name, platform, genre, publisher, developer, rating, videoGameDataset)

    # ==========branches==========#

    """
    print(videoGameDataset)
    print(yRevenue)
    print(yIndieAAAlabel)
    print(ySuccessLabels)
    """

    fakeVideoGameDF = pd.DataFrame(
        [[1, 1, year, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1, 1]],
        columns=videoGameDataset.columns)

    # print(fakeVideoGameDF)

    if model == "regressional":
        regressional(yRevenue, originalDataset, fakeVideoGameDF, videoGameDataset)

    elif model == "categorical":
        categorical(originalDataset, fakeVideoGameDF, videoGameDataset, yIndieAAAlabel, ySuccessLabels)

    elif model == "both":
        regressional(yRevenue, originalDataset, fakeVideoGameDF, videoGameDataset)
        categorical(originalDataset, fakeVideoGameDF, videoGameDataset, yIndieAAAlabel, ySuccessLabels)


if __name__ == "__main__":
    main()
