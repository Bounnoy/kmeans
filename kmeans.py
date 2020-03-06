# Bounnoy Phanthavong (ID: 973081923)
# Homework 4
#
# This is a machine learning program that uses a K-means Clustering to
# classify optical digits.
#
# This program was built in Python 3.

from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
import pickle
import time
import sys
import random

class KMeans:
    def __init__(self, train, test):
        self.trainData = train
        self.testData = test

    def calculate_euclidian(self, x, y):
        return np.sum(np.square(x-y))

    def train(self, kvalue):
        xs = len(self.trainData[0]) - 1         # Attributes in training data.
        rows = len(self.trainData)

        bestRun = 0
        bestMSE = 999999
        bestMSS = 0
        bestME = 0
        bestM = np.zeros((kvalue, xs))

        # Run training on the data 5 times and pick the best run.
        for runs in range(5):
            random.seed()

            # Initialize initial cluster points.
            mi = []
            for j in range(kvalue):
                val = random.randint(0, rows)
                while val in mi:
                    val = random.randint(0, rows)
                mi.append(val)

            # Change m to actual values instead of index.
            # m is a row of 64 inputs.
            m = np.zeros((kvalue, xs+1))
            for j in range(kvalue):
                m[j] = np.array([self.trainData[mi[j]]])

            d2 = np.zeros((rows, kvalue))

            iter = True
            count = 0

            while iter:
                count += 1
                # Calculate euclidian distance for each x and y in training data.
                members = {}
                for l in range(rows):
                    min = 0

                    for k in range(kvalue):
                        d2[l][k] = self.calculate_euclidian(self.trainData[l][:-1], m[k][:-1])

                        # Update min = index of smallest distance.
                        if d2[l][k] < d2[l][min]:
                            min = k
                        if d2[l][k] == d2[l][min]:
                            min = random.choice([k, min])

                    # Add index of point in members list by class.
                    if min in members:
                        members[min] = np.append(members[min], np.array([self.trainData[l]]), axis=0)
                    else:
                        members[min] = np.array([self.trainData[l]])

                same = 0 # Flag to check if m value didn't change.

                # Update centroids.
                # Note: It's possible there are no members for a certain class.
                for k in range(kvalue):
                    if k in members:
                        upval = members[k].sum(axis=0) / len(members[k])
                        if np.all(np.equal(upval, m[k])):
                            same += 1
                        m[k] = upval
                    else:
                        same += 1

                # If centroids don't change, don't iterate anymore.
                if same == kvalue:
                    iter = False

            me = 0
            mse = 0

            # Calculate average mean squared error.
            for key, value in members.items():
                most = 0

                # Calculate mean entropy.
                (classes, cnum) = np.unique(value[:,-1], return_counts=True)
                entropy = 0
                entotal = np.sum(cnum)
                for i in range(len(cnum)):
                    entropy += (cnum[i]/entotal) * math.log(cnum[i]/entotal)

                    # Check the most frequent class.
                    if cnum[i] > most:
                        m[key][-1] = classes[i]
                        most = cnum[i]
                    if cnum[i] == most:
                        m[key][-1] = random.choice([classes[i], m[key][-1]])

                entropy *= -1
                me += (entotal/rows) * entropy

                sum = 0

                # Calculate average mean squared error.
                for i in range(len(value)):
                    sum += self.calculate_euclidian(value[i][:-1], m[key][:-1])
                mse += (sum / len(value))
            mse /= kvalue

            mss = 0

            # Calculate mean squared separation.
            for i in range(len(m)):
                for j in range(len(m)):
                    j += i+1
                    if j < len(m):
                        mss += self.calculate_euclidian(m[i][:-1], m[j][:-1])
            mss /= (kvalue * (kvalue - 1) / 2)

            if mse < bestMSE:
                bestRun = runs
                bestMSE = mse
                bestMSS = mss
                bestME = me
                bestM = m

        print("Best run: #" + str(bestRun+1))
        print("Average MSE = " + str(bestMSE))
        print("MSS = " + str(bestMSS))
        print("Mean Entropy = " + str(bestME))

        # Calculate euclidian distance for each x and y in test data.
        trows = len(self.testData)
        td2 = np.zeros((trows, kvalue))
        correct = 0
        matrix = np.zeros((10, 10))

        for l in range(trows):
            min = 0

            for k in range(kvalue):
                td2[l][k] = self.calculate_euclidian(self.testData[l][:-1], m[k][:-1])

                # Update min = index of smallest distance.
                if td2[l][k] < td2[l][min]:
                    min = k
                if td2[l][k] == td2[l][min]:
                    min = random.choice([k, min])

            # If prediction is correct, increment correct counter.
            if int(m[min][-1]) == int(self.testData[l][-1]):
                correct += 1

            # Plot our data in the table.
            matrix[ int(m[min][-1]) ][ int(self.testData[l][-1]) ] += 1
            accuracy = (float(correct)/float(trows))

        print("Accuracy = " + str(accuracy))

        np.set_printoptions(suppress = True)
        print("\nConfusion Matrix")
        print(matrix, "\n")

        with open('results.csv', 'a') as csvFile:
            w = csv.writer(csvFile)
            w.writerow([])
            w.writerow(["Confusion Matrix"])
            for j in range(10):
                w.writerow(matrix[j,:])
            w.writerow(["Final Accuracy"] + [accuracy])
            w.writerow([])

        # Display images.
        for i in range(kvalue):
            plt.imshow(np.reshape(m[i][:-1], (8, 8)), cmap='Greys')
            plt.savefig("k"+str(kvalue)+"cluster"+str(i))

        return

if __name__ == '__main__':

    trainName = "optdigits/optdigits.train"
    testName = "optdigits/optdigits.test"

    pklTrain = Path(trainName + ".pkl")
    pklTest = Path(testName + ".pkl")
    fileTrain = Path(trainName)
    fileTest = Path(testName)

    if not fileTrain.exists():
        sys.exit(trainName + " not found")

    if not fileTest.exists():
        sys.exit(testName + " not found")

    if not pklTrain.exists():
        f1 = np.genfromtxt(trainName, delimiter=",")
        csv1 = open(trainName + ".pkl", 'wb')
        pickle.dump(f1, csv1)
        csv1.close()

    if not pklTest.exists():
        f2 = np.genfromtxt(testName, delimiter=",")
        csv2 = open(testName + ".pkl", 'wb')
        pickle.dump(f2, csv2)
        csv2.close()

    file1 = open(trainName + ".pkl", "rb")
    train = pickle.load(file1)
    file1.close()

    file2 = open(testName + ".pkl", "rb")
    test = pickle.load(file2)
    file2.close()

    km = KMeans(train, test)
    km.train(10)
    km.train(30)
