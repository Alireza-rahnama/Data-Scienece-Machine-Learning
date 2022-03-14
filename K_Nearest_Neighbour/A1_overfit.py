import sys
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sys import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
tsv_file = sys.argv[1]
raw_data = pd.read_csv(tsv_file, sep='\t')
#print(raw_data.describe())
featuresX=raw_data[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']]
classY=raw_data['class']


X1,X2,X3,X4,X5,X6,X7,X8,X9,X10=raw_data['X1'],raw_data['X2'],raw_data['X3'],raw_data['X4'],raw_data['X5'],raw_data['X6'],raw_data['X7'],raw_data['X8'],raw_data['X9'],raw_data['X10']
featurs=[X1,X2,X3,X4,X5,X6,X7,X8,X9,X10]


######################randomly seects 25 instances
testdata=raw_data.sample(25)

testData_features=testdata[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10']]
testData_output=testdata['class']



# split dataset by using function train_test_split()
# 100% training data


#Create KNN Classifier For number of neighbors k = {1, 3, 5, 7, 9}
#computing the scores/accuracy of each instance
k = [1, 3, 5, 7, 9]
train_scores1=[]
test_scores1=[]
for i in range(5):
    knn = KNeighborsClassifier(n_neighbors= k[i])

    # Train the model using the 100% of the data as the training set
    knn.fit(featuresX, classY)

    # Predict the response for the test dataset
    y_pred = knn.predict(testData_features)
    
    #training and test error is equal to 1-accuracy of each
    #training eror
    train_score=1-knn.score(featuresX,classY)
    #test eror
    test_score=1-knn.score(testData_features,testData_output)
    
    train_scores1.append(train_score)
    test_scores1.append(test_score)




 # Model Accuracy, how often is the classifier correct?
    print("the calculated Accuracy of the prediction with "+str(k[i])+" neighbors where %100 of data is used as training data is equal to", metrics.accuracy_score(testData_output, y_pred))

plt.plot(k,train_scores1, color='red', label="Training Error with 100% of data in training set")
plt.plot(k,test_scores1, color='blue', label="Testing Error with 100% of data in training set")
plt.xlabel('K values')
plt.ylabel('Error/Loss')
plt.title('Performace Under Varying K Values') 

print("..........................................................................")


### 75% training and 25% test data
X_train, X_test, y_train, y_test = train_test_split(featuresX, classY, test_size=25)
k = [1, 3, 5, 7, 9]
train_scores2=[]
test_scores2=[]
for i in range(5):
    knn = KNeighborsClassifier(n_neighbors=k[i])

    # Train the model using the training sets
    knn.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = knn.predict(X_test)
    #training and test error 1-accuracy
    train_scor=1-knn.score(X_train,y_train)
    test_scor=1-knn.score(X_test,y_test)
    train_scores2.append(train_scor)
    test_scores2.append(test_scor)

    # Model Accuracy, how often is the classifier correct?
    print("the calculated Accuracy of the prediction with "+str(k[i])+" neighbors where %75 of data is used as training data is equal to", metrics.accuracy_score(y_test, y_pred))
#ploting test error/trainingerror for varying k(no. of neighbours) values
plt.plot(k,train_scores2, color='yellow', label="Training Error with 75% of data in training set")
plt.plot(k,test_scores2, color='black', label="Testing Error with 75% of data in training set")
plt.xlabel('K values')
plt.ylabel('Error/Loss')
#plt.title('Performace Under Varying K Values when 75% instances in the data is used as the training data') 
plt.legend()
plt.show()





for featur in featurs:
    class0=[]
    class1=[]
    classes=[]
    for i in range(raw_data.shape[0]):
        for j in classY:
            if j==0:
                class0.append(featur[i])
            else:
                class1.append(featur[i])
    classes.append(class0)
    classes.append(class1)

    #ploting box plot
    fig = plt.figure(figsize =(10, 7))

    ax = fig.add_axes([0, 0, 1, 1])
    
    # Creating plot
    bp = ax.boxplot(classes,labels=["class 0","class 1"])
    
    # x-axis labels
    # ax.set_xticklabels(['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10'])
    # Adding title
    plt.title("box plot")
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticklabels(['class 0', 'class 1'])
    plt.show()
