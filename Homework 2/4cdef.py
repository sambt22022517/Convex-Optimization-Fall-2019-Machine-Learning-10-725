import cvxpy as cp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import time

groupLabels = []
with open('movies/groupLabelsPerRating.txt') as groupLabelsFile:
    labels = groupLabelsFile.read().strip().split(',')
    groupLabels = np.array([int(i) for i in labels])

J = [0]
setgroupLabels = set(groupLabels)
for i in setgroupLabels:
    J.append(J[-1] + sum(groupLabels == i))
J = np.array(J)

wJ = []
for j in range(len(J)-1):
    wJ.append(math.sqrt(J[j+1] - J[j]))
wJ = np.array(wJ)

trainLabels = []
with open('movies/trainLabels.txt') as trainLabelsFile:
    labels = trainLabelsFile.read().strip().split('\n')
    trainLabels = np.array([int(i) for i in labels])

trainRatings = []
with open('movies/trainRatings.txt') as trainRatingsFile:
    labels = trainRatingsFile.read().strip().split('\n')
    trainRatings = np.array([np.array([int(j) for j in i.split(',')]) for i in labels])

class PGD:
    def __init__(self, LAMBDA=5, T=1e-4):
        self.beta = np.zeros(len(groupLabels))
        self.X = trainRatings
        self.y = trainLabels
        self.LAMBDA = LAMBDA
        self.t = T
        self.J = J
        self.wJ = wJ
        
    def norm2(self, vector):
        sumVector = 0
        for i in vector:
            sumVector += i*i
        return math.sqrt(sumVector)
    
    def proximal(self, beta):
        result = np.zeros(len(beta))
        
        norm2BetaJ = []
        for j in range(len(self.J)-1):
            norm2BetaJ.append(self.norm2(beta[self.J[j]:self.J[j+1]]))
            
        for j in range(len(self.J)-1):
            for i in range(J[j], J[j+1]):
                result[i] = beta[i]
                if norm2BetaJ[j] > 0:
                    result[i] -= self.t * self.LAMBDA * self.wJ[j] * beta[i] / norm2BetaJ[j]
        
        return np.array(result)
    
    def gradient(self, beta):
        result = []
        
        XBeta = []
        for row in self.X:
            XBetai = 0
            for i in range(len(row)):
                XBetai += row[i] * beta[i]
            XBeta.append(XBetai)
        XBeta = np.array(XBeta)
        
        for k in range(len(beta)):
            gk = 0
            for i in range(len(self.y)):
                # start = time.time()
                eXBetai = math.exp(XBeta[i])
                add = self.X[i][k] * (eXBetai / (1 + eXBetai) - self.y[i])
                gk += add
                # print(time.time()-start)
            result.append(gk)
        
        return np.array(result)
    
    def proximal_gradient_descent(self, iteration=1000):
        for i in range(iteration):
            self.beta = self.proximal(self.beta - self.t * self.gradient(self.beta))

pdg = PGD()
pdg.proximal_gradient_descent(1000)
print(len(pdg.y))