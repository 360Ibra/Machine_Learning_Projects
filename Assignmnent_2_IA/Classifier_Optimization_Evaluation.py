from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import *
import time

# Pre Processing and Visualization
pd.set_option('display.max_rows', None)


def crossValidation(chosen_cls, features, target, sample_size,n,k=0,y=0,):
    features = features.drop("index", axis="columns")
    target = target.drop("index", axis="columns")
    # Slicing Notation to limit sample size
    # print(features[:50])
    # print(target)
    features = features.to_numpy()
    target = target.to_numpy()
    allResults = []
    allTimesTrain = []
    allTimesPredict = []
    mins = []
    maxes =  []

    kf = model_selection.KFold(n_splits=n, shuffle=True)
    for train_index, test_index in kf.split(features[:sample_size]):

        if chosen_cls == "Perceptron":
            print()
            print("*Perceptron Classifier Results*")
            print()
            start = time.time()
            clf = linear_model.Perceptron()
            clf.fit(features[train_index], target[train_index])
            end = time.time()
            allTimesTrain.append(end-start)
            print("Training Time Elapsed :", end - start)

            start = time.time()
            results = clf.predict(features[test_index])
            end = time.time()
            allTimesPredict.append(end-start)
            print()
            print("Prediction Time Elapsed :",end - start)
            print(results)
            # print(np.array(results))
            # print(target[test_index])
            # print(results[results != target[test_index].flatten()])

            allResults.append(metrics.accuracy_score(results, target[test_index].flatten()))
            print("Accuracy : ",metrics.accuracy_score(results, target[test_index].flatten()))
            print("Confusion Matrix :\n ", metrics.confusion_matrix(results, target[test_index].flatten()))



        elif chosen_cls == "DT":
            print()
            print("*Decision Tree Classifier Results*")
            print()
            start = time.time()
            clf = tree.DecisionTreeRegressor()
            clf.fit(features[train_index], target[train_index])
            end = time.time()
            allTimesTrain.append(end - start)
            print("Training Time Elapsed :", end - start)

            start = time.time()
            results = clf.predict(features[test_index])
            end = time.time()
            allTimesPredict.append(end - start)
            print("Prediction Time Elapsed :", end - start)

            # print(np.array(results))
            # print(target[test_index])
            # print(results[results != target[test_index].flatten()])

            allResults.append(metrics.accuracy_score(results, target[test_index].flatten()))
            print("Accuracy : ", metrics.accuracy_score(results, target[test_index].flatten()))
            print("Confusion Matrix :\n ", metrics.confusion_matrix(results, target[test_index].flatten()))

        elif chosen_cls == "KNN":
            print()
            print("*K-Nearest Neighbour Classifier Results*")
            print()

            start = time.time()
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf.fit(features[train_index], target[train_index])
            end = time.time()
            allTimesTrain.append(end - start)
            print("Training Time Elapsed :", end - start)

            start = time.time()
            results = clf.predict(features[test_index])
            end = time.time()
            allTimesPredict.append(end - start)
            print("Prediction Time Elapsed :", end - start)

            # print(np.array(results))
            # print(target[test_index])
            # print(results[results != target[test_index].flatten()])

            allResults.append(metrics.accuracy_score(results, target[test_index].flatten()))
            print("Accuracy : ", metrics.accuracy_score(results, target[test_index].flatten()))
            print("Confusion Matrix :\n ", metrics.confusion_matrix(results, target[test_index].flatten()))
            print("Testing the amount of k",k)



        elif chosen_cls == "SVM":
            print()
            print("*SVM Classifier Results*")
            print()

            start = time.time()
            clf = svm.SVC(gamma=y)
            clf.fit(features[train_index], target[train_index])
            end = time.time()
            allTimesTrain.append(end - start)
            print("Training Time Elapsed :", end - start)

            start = time.time()
            results = clf.predict(features[test_index])
            end = time.time()
            allTimesPredict.append(end - start)
            print("Prediction Time Elapsed :", end - start)

            # print(np.array(results))
            # print(target[test_index])
            # print(results[results != target[test_index].flatten()])

            allResults.append(metrics.accuracy_score(results, target[test_index].flatten()))
            print("Accuracy : ", metrics.accuracy_score(results, target[test_index].flatten()))
            print("Confusion Matrix :\n ", metrics.confusion_matrix(results, target[test_index].flatten()))
            print("Testing the amount of y",y)

    # Average,Max,Min Accuracy Results

    print("Average Accuracy : ", np.mean(allResults))
    print("Max Accuracy : ",max(allResults))
    print("Min Accuracy : ", min(allResults))
    print()
    # Average,Max,Min Train Time Results
    print()
    print("Average Train Time : ", np.mean(allTimesTrain))
    print("Max Train Time : ", max(allTimesTrain))
    print("Min Train Time : ", min(allTimesTrain))
    print()
    # Average,Max,Min Prediction Time Results
    print()
    print("Average Prediction Time : ", np.mean(allTimesPredict))
    print("Max Prediction Time : ", max(allTimesPredict))
    print("Min Prediction Time : ", min(allTimesPredict))
    print()

    # Appending Max and Mins for plotting
    mins.append(min(allTimesTrain))
    maxes.append(max(allTimesTrain))
    print("should be a list ",maxes)

    return np.mean(allTimesTrain),np.mean(allTimesPredict),max(allTimesTrain),min(allTimesTrain),max(allTimesPredict),min(allTimesPredict)
    # print(allResults)
    # print(allTimesTrain)
    # print(allTimesPredict)



def Task1(df):
    # Conditions for separating sneakers , sandals and ankle boots
    # Into a dataframe
    sandal_condition = df["label"] == 5
    boots_condition = df["label"] == 9
    sneaker_condition = df["label"] == 7
    sandals_sneakers_boots = df[sandal_condition | boots_condition | sneaker_condition]
    # Separating the dataset labels from the feature vectors
    target = sandals_sneakers_boots["label"]
    features = sandals_sneakers_boots.drop("label",axis="columns")

    # print(sandals_sneakers_boots)

    return  sandals_sneakers_boots, target, pd.DataFrame(features)
df = pd.read_csv("fashion-mnist_train.csv")

returned ,target,features = Task1(df)
features = features.reset_index()
target = target.reset_index()

# Looping through column rows left to right
# print(features.iloc[500,:])
img = []
shoes = [features.iloc[1,:],features.iloc[2,:],features.iloc[500,:]]
for shoe in shoes:
    for i in shoe:
        img.append(i)

    img.pop(0)
    # print(img)





    img = np.array(img)
    "Reshaping the image into 28 by 28" \
    "So that the images of shoes can be visualize"
    plt.imshow(img.reshape(28,28))
    plt.show()
    img = []






"Error that I learned from 1 "
"When the you get the error expected indices to be 1 dimenstional but got 2"
"Use flatten on the object which 2d to change it to 1 dimension"
'print(f"TARTGET TEST - {target[test_index].flatten().shape}")'
'print(f"RESULTS - {np.array(results).shape}")'
"print(target[train_index],features[train_index])"

t = []
p = []
tmaxes = []
tmins = []
pmaxes = []
pmins = []

# samples = [900,1800,2700,3600,4500,5400,6000,7200,8100,9000,9900,10800,11700,12600,13500,14400,15300,16200,17100,18000]

samples = [100,400,800,2000,4000,6000,8000,12000,14000,18000]

# samples = [100,200,400,800,1000,2000,3000,4000,5000]
ys = [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]
# ys = [2e-4,2e-5,2e-6,2e-7,2e-8,2e-9]
ks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
# classifiers = ["Perceptron","DT","KNN","SVM"]
size = len(samples)


# Uncomment One at a time
for s in samples:
    print("CURRENT SAMPLE SIZE ->%d",s)
    trainTimes ,predTimes,trainMax,trainMin,predMax,predMin = crossValidation("Perceptron",features,target,s,n=50)
    # trainTimes, predTimes, trainMax, trainMin, predMax, predMin = crossValidation("DT",features,target,s,n=12)
    # for k in ks:
    #     trainTimes ,predTimes,trainMax,trainMin,predMax,predMin= crossValidation("KNN", features, target, s, k=k,n=size)
    # for y in ys:
    #     trainTimes ,predTimes,trainMax,trainMin,predMax,predMin= crossValidation("SVM",features,target,s,n=size,y=y)
    # trainTimes ,predTimes,trainMax,trainMin,predMax,predMin= crossValidation("KNN", features, target, s, k=2,n=3)
    # trainTimes ,predTimes,trainMax,trainMin,predMax,predMin= crossValidation("SVM",features,target,s,n=size,y=1e-6)
    t.append(trainTimes)
    p.append(predTimes)
    tmaxes.append(trainMax)
    tmins.append(trainMin)
    pmaxes.append(predMax)
    pmins.append(predMin)




plt.plot(samples,t,label="Train")
plt.plot(samples,p,label="Predict")
plt.plot(samples,tmaxes,label="Train Time Maxes")
plt.plot(samples,tmins,label="Train Time Mins")
plt.plot(samples,pmaxes,label="Prediction Time Maxes")
plt.plot(samples,pmins,label="Prediction Time Mins")
plt.legend()
plt.show()

# print(t)
# print(p)

# print(tmaxes)
# print(tmins)
# print(pmaxes)
# print(pmins)
#

# trainTimes ,predTimes,trainMax,trainMin,predMax,predMin = crossValidation("Perceptron",features,target,18000,n=5)
# trainTimes ,predTimes,trainMax,trainMin,predMax,predMin = crossValidation("DT",features,target,18000,n=5)
# trainTimes ,predTimes,trainMax,trainMin,predMax,predMin = crossValidation("KNN", features, target, 18000, k=8,n=5)
# trainTimes ,predTimes,trainMax,trainMin,predMax,predMin = crossValidation("SVM",features,target,18000,n=5,y=1e-3)


# Parameterise the number of samples
# print(features[:60])
# print(target[:60])