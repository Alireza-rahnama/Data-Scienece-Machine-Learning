import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
# import seaborn as sns
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors, datasets, preprocessing

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import scale
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
from sklearn.linear_model import *
import sys

from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
tsv_file1 = sys.argv[1]
tsv_file2=sys.argv[2]
training_data = pd.read_csv(tsv_file1, sep='\t')
test_data = pd.read_csv(tsv_file2, sep='\t')


# training_data = pd.read_csv('A3_TrainData.tsv', sep='\t')
# # print(training_data.shape)
# test_data= pd.read_csv('A3_TestData.tsv', sep='\t')
# # print(training_data.head)

X_train=training_data[["SS","PromoterDistance","DistTerm","Distance","sameStrand","DownDistance","sameDownStrand","AAAA/TTTT","AAAC/GTTT","AAAG/CTTT","AAAT/ATTT","AACA/TGTT","AACC/GGTT","AACG/CGTT","AACT/AGTT","AAGA/TCTT","AAGC/GCTT","AAGG/CCTT","AAGT/ACTT","AATA/TATT","AATC/GATT","AATG/CATT","AATT/AATT","ACAA/TTGT","ACAC/GTGT","ACAG/CTGT","ACAT/ATGT","ACCA/TGGT","ACCC/GGGT","ACCG/CGGT","ACCT/AGGT","ACGA/TCGT","ACGC/GCGT","ACGG/CCGT","ACGT/ACGT","ACTA/TAGT","ACTC/GAGT","ACTG/CAGT","AGAA/TTCT","AGAC/GTCT","AGAG/CTCT","AGAT/ATCT","AGCA/TGCT","AGCC/GGCT","AGCG/CGCT","AGCT/AGCT","AGGA/TCCT","AGGC/GCCT","AGGG/CCCT","AGTA/TACT","AGTC/GACT","AGTG/CACT","ATAA/TTAT","ATAC/GTAT","ATAG/CTAT","ATAT/ATAT","ATCA/TGAT","ATCC/GGAT","ATCG/CGAT","ATGA/TCAT","ATGC/GCAT","ATGG/CCAT","ATTA/TAAT","ATTC/GAAT","ATTG/CAAT","CAAA/TTTG","CAAC/GTTG","CAAG/CTTG","CACA/TGTG","CACC/GGTG","CACG/CGTG","CAGA/TCTG","CAGC/GCTG","CAGG/CCTG","CATA/TATG","CATC/GATG","CATG/CATG","CCAA/TTGG","CCAC/GTGG","CCAG/CTGG","CCCA/TGGG","CCCC/GGGG","CCCG/CGGG","CCGA/TCGG","CCGC/GCGG","CCGG/CCGG","CCTA/TAGG","CCTC/GAGG","CGAA/TTCG","CGAC/GTCG","CGAG/CTCG","CGCA/TGCG","CGCC/GGCG","CGCG/CGCG","CGGA/TCCG","CGGC/GCCG","CGTA/TACG","CGTC/GACG","CTAA/TTAG","CTAC/GTAG","CTAG/CTAG","CTCA/TGAG","CTCC/GGAG","CTGA/TCAG","CTGC/GCAG","CTTA/TAAG","CTTC/GAAG","GAAA/TTTC","GAAC/GTTC","GACA/TGTC","GACC/GGTC","GAGA/TCTC","GAGC/GCTC","GATA/TATC","GATC/GATC","GCAA/TTGC","GCAC/GTGC","GCCA/TGGC","GCCC/GGGC","GCGA/TCGC","GCGC/GCGC","GCTA/TAGC","GGAA/TTCC","GGAC/GTCC","GGCA/TGCC","GGCC/GGCC","GGGA/TCCC","GGTA/TACC","GTAA/TTAC","GTAC/GTAC","GTCA/TGAC","GTGA/TCAC","GTTA/TAAC","TAAA/TTTA","TACA/TGTA","TAGA/TCTA","TATA/TATA","TCAA/TTGA","TCCA/TGGA","TCGA/TCGA","TGAA/TTCA","TGCA/TGCA","TTAA/TTAA"]]
Y_train=training_data[["Class"]]
X_test=test_data[["SS","PromoterDistance","DistTerm","Distance","sameStrand","DownDistance","sameDownStrand","AAAA/TTTT","AAAC/GTTT","AAAG/CTTT","AAAT/ATTT","AACA/TGTT","AACC/GGTT","AACG/CGTT","AACT/AGTT","AAGA/TCTT","AAGC/GCTT","AAGG/CCTT","AAGT/ACTT","AATA/TATT","AATC/GATT","AATG/CATT","AATT/AATT","ACAA/TTGT","ACAC/GTGT","ACAG/CTGT","ACAT/ATGT","ACCA/TGGT","ACCC/GGGT","ACCG/CGGT","ACCT/AGGT","ACGA/TCGT","ACGC/GCGT","ACGG/CCGT","ACGT/ACGT","ACTA/TAGT","ACTC/GAGT","ACTG/CAGT","AGAA/TTCT","AGAC/GTCT","AGAG/CTCT","AGAT/ATCT","AGCA/TGCT","AGCC/GGCT","AGCG/CGCT","AGCT/AGCT","AGGA/TCCT","AGGC/GCCT","AGGG/CCCT","AGTA/TACT","AGTC/GACT","AGTG/CACT","ATAA/TTAT","ATAC/GTAT","ATAG/CTAT","ATAT/ATAT","ATCA/TGAT","ATCC/GGAT","ATCG/CGAT","ATGA/TCAT","ATGC/GCAT","ATGG/CCAT","ATTA/TAAT","ATTC/GAAT","ATTG/CAAT","CAAA/TTTG","CAAC/GTTG","CAAG/CTTG","CACA/TGTG","CACC/GGTG","CACG/CGTG","CAGA/TCTG","CAGC/GCTG","CAGG/CCTG","CATA/TATG","CATC/GATG","CATG/CATG","CCAA/TTGG","CCAC/GTGG","CCAG/CTGG","CCCA/TGGG","CCCC/GGGG","CCCG/CGGG","CCGA/TCGG","CCGC/GCGG","CCGG/CCGG","CCTA/TAGG","CCTC/GAGG","CGAA/TTCG","CGAC/GTCG","CGAG/CTCG","CGCA/TGCG","CGCC/GGCG","CGCG/CGCG","CGGA/TCCG","CGGC/GCCG","CGTA/TACG","CGTC/GACG","CTAA/TTAG","CTAC/GTAG","CTAG/CTAG","CTCA/TGAG","CTCC/GGAG","CTGA/TCAG","CTGC/GCAG","CTTA/TAAG","CTTC/GAAG","GAAA/TTTC","GAAC/GTTC","GACA/TGTC","GACC/GGTC","GAGA/TCTC","GAGC/GCTC","GATA/TATC","GATC/GATC","GCAA/TTGC","GCAC/GTGC","GCCA/TGGC","GCCC/GGGC","GCGA/TCGC","GCGC/GCGC","GCTA/TAGC","GGAA/TTCC","GGAC/GTCC","GGCA/TGCC","GGCC/GGCC","GGGA/TCCC","GGTA/TACC","GTAA/TTAC","GTAC/GTAC","GTCA/TGAC","GTGA/TCAC","GTTA/TAAC","TAAA/TTTA","TACA/TGTA","TAGA/TCTA","TATA/TATA","TCAA/TTGA","TCCA/TGGA","TCGA/TCGA","TGAA/TTCA","TGCA/TGCA","TTAA/TTAA"]]



# grid searching key hyperparametres for logistic regression
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
# # define models and parameters
# model= LogisticRegression()
# solvers = ['newton-cg', 'lbfgs', 'liblinear']
# penalty = ['l2']
# c_values = [100, 10, 1.0, 0.1, 0.01]
# # define grid search
# grid = dict(solver=solvers,penalty=penalty,C=c_values)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train,Y_train.values.flatten())
# # summarize results
# print("Best performance of logistic regression: %f using %s HyperParameters" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# # for mean, stdev, param in zip(means, stds, params):
# #     print("%f (%f) with: %r" % (mean, stdev, param))

# # ##############Best: 0.795502 using {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}

# #now we use the optimum hyperparameters to construct our logistic regression
# log_regression_model= LogisticRegression(C= 100, penalty= 'l2', solver= 'newton-cg')
# log_regression_model.fit(X_train,Y_train.values.flatten())



# #############  grid searching key hyperparametres for ridge classifier
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import RidgeClassifier
# ## define models and parameters
# model = RidgeClassifier()
# alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# # define grid search
# grid = dict(alpha=alpha)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train,Y_train.values.flatten())
# # summarize results
# print("Best performance of ridge classifier: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# # for mean, stdev, param in zip(means, stds, params):
# #     print("%f (%f) with: %r" % (mean, stdev, param))
# ############## Best performance of ridge classifier: 0.795502 using {'alpha': 0.1}Best performance of ridge classifier: 0.795502 using {'alpha': 0.1}


# ############ grid searching key hyperparametres for KNeighborsClassifier
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier

# # define models and parameters
# model = KNeighborsClassifier()
# n_neighbors = range(1, 21, 2)
# weights = ['uniform', 'distance']
# metric = ['euclidean', 'manhattan', 'minkowski']
# # define grid search
# grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train,Y_train.values.flatten())
# # summarize results
# print("Best performance obtained from KNN classifier: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# # for mean, stdev, param in zip(means, stds, params):
# #     print("%f (%f) with: %r" % (mean, stdev, param))
# ########## Best performance obtained from KNN classifier: 0.856206 using {'metric': 'manhattan', 'n_neighbors': 19, 'weights': 'distance'}



##########grid searching key hyperparameters for RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# # define models and parameters
# model = RandomForestClassifier()
# #no_estimator is the number f trees in the forest
# n_estimators = [10, 100, 1000]
# max_features = ['sqrt', 'log2']
# # define grid search
# grid = dict(n_estimators=n_estimators,max_features=max_features)
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
# grid_result = grid_search.fit(X_train,Y_train.values.flatten())
# # summarize results
# print("Best performance of RandomForestClassifier: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# #make the table with the CV results for the best model(random forest) method with mean and standard generated below
# # Standard deviation indicating the model selected to generate the final model
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("mean: %f StandardDeviation:(%f) with: %r" % (mean, stdev, param))


########## Best performance of RandomForestClassifier: 0.887779 using {'max_features': 'sqrt', 'n_estimators': 1000}



# roc curve and auc 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X_train, Y_train, test_size=0.6, random_state=2)
# # fit a model
model = RandomForestClassifier(max_features= 'sqrt', n_estimators= 1000)
model.fit(trainX, trainy.values.flatten())
model2=LogisticRegression(C= 100, penalty= 'l2', solver= 'newton-cg')
model2.fit(trainX, trainy.values.flatten())
model3=KNeighborsClassifier(metric= 'manhattan', n_neighbors= 19, weights= 'distance')
model3.fit(trainX, trainy.values.flatten())
model4=RidgeClassifier(alpha= 0.1)
model4.fit(trainX, trainy.values.flatten())
# # generate a no skill prediction (majority class)
# ns_probs = [0 for _ in range(len(testy))]

# predict probabilities
random_forest_probs = model.predict_proba(testX)
logistic_regression_probs=model2.predict_proba(testX)
KNeighbors_classifier_probs=model3.predict_proba(testX)
ridge_classifier_probs=model4._predict_proba_lr(testX)
# keep probabilities for the positive outcome only
random_forest_probs = random_forest_probs[:, 1]
logistic_regression_probs=logistic_regression_probs[:, 1]
KNeighbors_classifier_probs=KNeighbors_classifier_probs[:, 1]
ridge_classifier_probs=ridge_classifier_probs[:, 1]
# calculate scores

random_forest_auc = roc_auc_score(testy, random_forest_probs)
logistic_regression_auc=roc_auc_score(testy, logistic_regression_probs)
KNeighbors_classifier_auc=roc_auc_score(testy,KNeighbors_classifier_probs)
ridge_classifier_auc=roc_auc_score(testy,ridge_classifier_probs)
# summarize scores
# print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Random Forest: AUROC =%.3f' % (random_forest_auc))
print('Logistic Regression: AUROC =%.3f' % (logistic_regression_auc))
print('KNN: AUROC =%.3f' % (KNeighbors_classifier_auc))
print('Ridge Classifier: AUROC =%.3f' % (ridge_classifier_auc))
# calculate roc curves

random_forest_fpr, random_forest_tpr, _ = roc_curve(testy, random_forest_probs)
logistic_regression_fpr, logistic_regression_tpr, _ = roc_curve(testy, logistic_regression_probs)
KNeighbors_classifier_fpr, KNeighbors_classifier_tpr, _=roc_curve(testy, KNeighbors_classifier_probs)
ridge_classifier_fpr,ridge_classifier_tpr, _=roc_curve(testy, ridge_classifier_probs)
# plot the roc curve for the model
pyplot.plot(logistic_regression_fpr, logistic_regression_tpr, linestyle='--', label=('Logistic Regression with AUROC =%.3f' % (logistic_regression_auc)))
pyplot.plot(random_forest_fpr, random_forest_tpr, marker='.', label=('Random Forest with AUROC =%.3f' % (random_forest_auc)))
pyplot.plot(KNeighbors_classifier_fpr, KNeighbors_classifier_tpr, marker='.', label=('KNeighbors_classifier with AUROC =%.3f' % (KNeighbors_classifier_auc)))
pyplot.plot(ridge_classifier_fpr,ridge_classifier_tpr, marker='.', label=('ridge_classifier with AUROC =%.3f' % (ridge_classifier_auc)))

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



# precision-recall curve and f1 for an imbalanced dataset
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(X_train, Y_train, test_size=0.6, random_state=2)
# # fit a model
model = RandomForestClassifier(max_features= 'sqrt', n_estimators= 1000)
model.fit(trainX, trainy.values.flatten())
model2=LogisticRegression(C= 100, penalty= 'l2', solver= 'newton-cg')
model2.fit(trainX, trainy.values.flatten())
model3=KNeighborsClassifier(metric= 'manhattan', n_neighbors= 19, weights= 'distance')
model3.fit(trainX, trainy.values.flatten())
model4=RidgeClassifier(alpha= 0.1)
model4.fit(trainX, trainy.values.flatten())
# predict probabilities
random_forest_probs = model.predict_proba(testX)
logistic_regression_probs=model2.predict_proba(testX)
KNeighbors_classifier_probs=model3.predict_proba(testX)
ridge_classifier_probs=model4._predict_proba_lr(testX)
# keep probabilities for the positive outcome only
random_forest_probs = random_forest_probs[:, 1]
logistic_regression_probs=logistic_regression_probs[:, 1]
KNeighbors_classifier_probs=KNeighbors_classifier_probs[:, 1]
ridge_classifier_probs=ridge_classifier_probs[:, 1]
# predict class values
yhat = model.predict(testX)
yhat2=model2.predict(testX)
yhat3=model3.predict(testX)
yhat4=model4.predict(testX)

# calculate precision and recall for each threshold
random_forest_precision, random_forest_recall, _ = precision_recall_curve(testy, random_forest_probs)
KNN_precision, knn_recall, _ = precision_recall_curve(testy, KNeighbors_classifier_probs)
lr_precision, lr_recall, _ = precision_recall_curve(testy, logistic_regression_probs)
ridge_precision, ridge_recall, _ = precision_recall_curve(testy, ridge_classifier_probs)
# calculate scores
random_forest_f1, random_forest_auc = f1_score(testy, yhat), auc(random_forest_recall, random_forest_precision)
lr_f1, lr_auc = f1_score(testy, yhat2), auc(lr_recall, lr_precision)
KNN_f1, KNN_auc = f1_score(testy, yhat3), auc(knn_recall, KNN_precision)
ridge_f1, ridge_auc = f1_score(testy, yhat4), auc(ridge_recall, ridge_precision)

# summarize scores
print('Random Forest: f1=%.3f auc=%.3f' % (random_forest_f1, random_forest_auc))
print('Logistic Regression: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
print('KNN: f1=%.3f auc=%.3f' % (KNN_f1, KNN_auc))
print('Ridge Classifier: f1=%.3f auc=%.3f' % (ridge_f1, ridge_auc))
# plot the precision-recall curves

pyplot.plot( knn_recall, KNN_precision, marker='*', label='KNN')
pyplot.plot(random_forest_recall, random_forest_precision, marker='.', label='Random Forest')
pyplot.plot( lr_recall, lr_precision, linestyle='--', label='Logistic Classifier')
pyplot.plot(ridge_recall, ridge_precision,  marker='.', label='Rridge Classifier')


# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()



import numpy as np
# # fit a model
model = RandomForestClassifier(max_features= 'sqrt', n_estimators= 1000)
model.fit(X_train, Y_train.values.flatten())
# predict class label
Y_test=model.predict(X_test)
# predict probabilities
probs = model.predict_proba(X_test)
probs = probs[:, 1]
file1= open('A3_predictions_group7.txt' ,'w')
# print(probs)
# # predicts the class label
# for num in Y_test:
#     file1.write(str(num))
#     file1.write("\n")
# file1.close()

# # predicts the liklihood of belonging to  class, The larger the number the more likely the 
##instance is to belong to class 1
for num in probs:
    file1.write(str(num))
    file1.write("\n")
file1.close()



# create an imbalanced dataset
from numpy import unique
from sklearn.datasets import make_classification

# summarize dataset
classes = unique(Y_test)
total = len(Y_test)
for c in classes:
	n_examples = len(Y_test[Y_test==c])
	percent = n_examples / total * 100
	print('> Class=%d : %d/%d (%.1f%%)' % (c, n_examples, total, percent))

