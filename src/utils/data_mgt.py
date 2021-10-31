import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(validation_datasize):
    data = pd.read_csv('churn.csv')

    return data


def splitting_data(data,validation_datasize):
    X = data.drop(['Churn'],axis=1)
    y = data['Churn']
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(X,y,test_size=0.3,random_state=164)
     
    ## Splitting into X_train and X_valids
    
    X_valid , X_train = X_train_full[:validation_datasize] , X_train_full[validation_datasize:]
    y_valid , y_train = y_train_full[:validation_datasize] , y_train_full[validation_datasize:]
    
    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test) 

