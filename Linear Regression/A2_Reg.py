import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors, datasets, preprocessing
import statsmodels.api as sm

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.linear_model import *
import sys

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

###########################################################################33
# Read data from file 'filename.csv'
# (in the same directory that your python process is based)

tsv_file1 = sys.argv[1]
tsv_file2=sys.argv[2]
training_data = pd.read_csv(tsv_file1, sep='\t')

#training_data = pd.read_csv(r'C:\Users\shiraz computer\Desktop\MUN Bachelor courses\Winter2022\machineLearning\Assignment2\A2_TrainData.tsv', sep='\t')
# print(training_data.head)


X_train=training_data[['X1','X2','X3','X4','X5','X6','X7','X8']]
y_train=training_data[['Y']]


# step-1: create a cross-validation scheme
folds = KFold(n_splits = 10, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 9))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
rfe = RFE(lm)             

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=False)  


lr= model_cv.fit(X_train,y_train)
y_predict=lr.predict(X_train)
print("The coefficient of determination(r squared) obtained from Linear Regression:\n")
######score here returns The coefficient of determination(r squared) the closer to 1 the better model
print(lr.score(X_train,y_train),"\n") 
##print(pd.DataFrame(model_cv.cv_results_))
print("best model:",model_cv.best_estimator_)
print("best parameters: ",model_cv. best_params_)
print("\n")


alphas = np.logspace(-4, -0.5, 30)
tuned_parameters = [{"alpha": alphas}]

#perform grid search
#  specify model
lasso = Lasso(random_state=0, max_iter=10000)

# call GridSearchCV()
m2_cv = GridSearchCV(estimator = lasso, 
                        param_grid = tuned_parameters, 
                        scoring= 'r2', 
                        cv = 10, 
                        return_train_score=False,
                        verbose= 1)  


lasso_model= m2_cv.fit(X_train,y_train)



print("The coefficient of determination(r squared) obtained from Lasso:\n")
######score here returns The coefficient of determination(r squared) the closer to 1 the better model
print(lasso_model.score(X_train,y_train),"\n") 
##print(pd.DataFrame(m2_cv.cv_results_))
print("best model:",lasso_model.best_estimator_)
print("best parameters: ",lasso_model. best_params_)
print("\n")



from sklearn.ensemble import RandomForestRegressor
param_grid = {
                 'n_estimators': [1,2,3,4,5,6,7,8],
                 'max_depth': [2, 5, 7, 9]
             }

regr = RandomForestRegressor()
m3_cv = GridSearchCV(estimator = regr, 
                        param_grid = param_grid, 
                        scoring= 'r2', 
                        cv = 10, 
                        return_train_score=False,
                        verbose= 1
                        )  


forest_model= m3_cv.fit(X_train,y_train.values.flatten())



print("The coefficient of determination(r squared) obtained from RandomForestRegressor:\n")
######score here returns The coefficient of determination(r squared) the closer to 1 the better model
print(forest_model.score(X_train,y_train),"\n") 
##print(pd.DataFrame(m3_cv.cv_results_))
print("best model:",m3_cv.best_estimator_)
print("best parameters: ",m3_cv. best_params_)
print("\n")


#####based on the performance of each method, forestregression is the best
###with The coefficient of determination(r squared) obtained from RandomForestRegressor:0.9651454304937929
# best model: RandomForestRegressor(max_depth=9, n_estimators=8, random_state=0)
# best parameters:  {'max_depth': 9, 'n_estimators': 8}
##############################################################################3
#test_data = pd.read_csv(r'C:\Users\shiraz computer\Desktop\MUN Bachelor courses\Winter2022\machineLearning\Assignment2\A2_TestData.tsv', sep='\t')
# print(test_data.head)
# print(test_data.shape)


#read the test data and fir and predict the outputs
test_data = pd.read_csv(tsv_file2, sep='\t')
X_test=test_data[['X1','X2','X3','X4','X5','X6','X7','X8']]
best_model_selected=RandomForestRegressor(max_depth=9, n_estimators=8, random_state=0)
best_model_selected.fit(X_train,y_train.values.flatten())
training_prediction= best_model_selected.predict(X_train)
predictions=best_model_selected.predict(X_test)
# print("size of prediction",predictions.shape)
file1= open('A2_predictions_group6.txt' ,'w')
for num in predictions:
    file1.write(str(num))
    file1.write("\n")
file1.close()

# f2=open('A2_predictions_group6.txt' ,'r')
# print(f2.read())


# print(training_prediction,training_prediction.shape)
y_train_predicted= pd.DataFrame(training_prediction)

residuals_squared=np.square(y_train_predicted - y_train)
rss=np.sum(residuals_squared)

print("best model RSS is: ",rss)
