import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import math

trainX="train_inputs.txt"#training data file
trainY="train_pass_success.txt"#training data success/failure

testX="test_inputs.txt"#test data file
testY="test_pass_success.txt"#test data success/failure

savename="LogR_results.txt"

def calc_xP_LogR(trainX,trainY,testX,testY):#calculate expected passes from training and test input data

    X_train = load_data_from_file(trainX)  # get regression inputs
    y_train = load_data_from_file(trainY)  # Success or fail data

    X_test = load_data_from_file(testX)  # get regression inputs
    y_test = load_data_from_file(testY)  # Success or fail data

    y_naive = np.empty(len(y_test))
    y_naive.fill(np.mean(y_train))#Calculate naive xPass based on average pass completion rate
    print "Naive data RMSE:",math.sqrt(mean_squared_error(y_test, y_naive))

    xP_LogR = LogisticR(X_train, y_train, X_test)#xP calculated via logistic regression on training data set
    print "RMSE using Logistic Regression:",math.sqrt(mean_squared_error(y_test, xP_LogR))
    np.savetxt(savename, xP_LogR, delimiter=',')  # save text file with results

def LogisticR(X, y, pred_data):#logistic regression function
    clf = linear_model.LogisticRegression()  # define logistic regression parameters
    clf.fit(X, y)  # run logistic regression model on training data set
    y_pred = clf.predict_proba(pred_data)[:, 1]# Column 1 is probability of success
    return y_pred

def load_data_from_file(file):#load data from file
    #file is defined below
    f = open(file, 'r')
    data = np.genfromtxt(f, delimiter=',')#assumes delimiter is a comma
    data = np.delete(data, 0, 0)  # Erases the first row (i.e. the header)
    f.close()
    return data
