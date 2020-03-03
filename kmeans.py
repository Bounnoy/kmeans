# Bounnoy Phanthavong (ID: 973081923)
# Homework 4
#
# This is a machine learning program that uses a K-means Clustering to
# classify optical digits.
#
# This program was built in Python 3.

from pathlib import Path
import math
import numpy as np
import csv
import pickle
import time
import sys
import random

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

    print("Training rows/cols:", len(train), ",", len(train[0]))
    print("Testing rows/cols:", len(test), ",", len(test[0]))
