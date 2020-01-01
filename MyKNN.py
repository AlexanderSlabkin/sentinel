import numpy as np

from scipy import spatial

from sklearn.neighbors.dist_metrics import DistanceMetric
from sklearn.metrics.pairwise import euclidean_distances

import math


def distance(X, Y):
    sum = 0
    for i in range(len(X)):
        sum += math.sqrt((X[i]-Y[i])**2)#abs(X[i]-Y[i])
    return math.sqrt(sum)

distance1 = lambda a,b: np.sqrt(np.sum((a-b)**2))
distance2 = euclidean_distances
distance3 = spatial.distance.euclidean
def vote(Neighbors, Votes):
    Classes = np.unique(Neighbors)
    Res = list(range(len(Classes)))
    for index, i in enumerate(Neighbors):
        for num, j in enumerate(Classes):
            if i == j:
                Res[num] += 1/(Votes[index]**2)
    return Classes[Res.index(max(Res))]



class MyKNN:
    def __init__(self, NNeighbors=3, Etalons = [], Marks = [], Metric='minkowski'):
        self.NNeighbors = NNeighbors
        self.Metric = Metric
        self.Etalons = Etalons
        self.Marks = Marks
        self.N = 0

    def classify(self, X):
        N = len(self.Etalons)
        Classification = list(range(len(X)))
        Distances = list(range(N))
        if self.NNeighbors > 1:
            for num, i in enumerate(X):
                for index, j in enumerate(self.Etalons):
                    Distances[index] = distance(i,j)
                Mins, Neighbors = [], []
                Max = max(Distances)
                for j in range(self.NNeighbors):
                    Index = Distances.index(min(Distances))
                    Mins.append(Distances[Index])
                    Neighbors.append(self.Marks[Index])
                    Distances[Index] = Max
                    Classification[num] = vote(Neighbors, Mins)
        else:
            for num, i in enumerate(X):
                for index, j in enumerate(self.Etalons):
                    Distances[index] = distance(i,j)
                Classification[num] = self.Marks[Distances.index(min(Distances))]


        self.N += 1
        return Classification
