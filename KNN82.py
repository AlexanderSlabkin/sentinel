#!/usr/bin/env python3
import numpy as np
from pandas import DataFrame
import argparse

import datetime
from time import perf_counter
from sklearn import neighbors
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import recall_score, accuracy_score

import configparser
import sys

from snappy import Product, ProductData, ProductIO, ProductUtils, FlagCoding

config = configparser.ConfigParser()
config.read('KNN8.cfg')

newConfig = configparser.ConfigParser()
newConfig.read('KNN8.cfg')

parser = argparse.ArgumentParser(description='Classify objects in multispectral space image. Operates with .dim files')
parser.add_argument('-i', '--infile', type = str, dest = 'File', default = config['DEFAULT']['file'],\
                                    help = ' .dim file')
parser.add_argument('-o', '--outfile', type = str, dest = 'OutFile', default = config['DEFAULT']['outfile'],\
                                    help = ' .dim file')
parser.add_argument('-p', '--points', type = str, dest = 'PointsIn', default = config['DEFAULT']['points source'],\
                                    help = ' .cfg file')
parser.add_argument('-c', '--changecfg', dest = 'ChangeCfg', default = False,\
                                    help = 'If True overwrites config files with provided parameters',\
                                    action="store_true")
parser.add_argument('-l', '--log', dest = 'IfLog', default = True,\
                                    help = 'Defines if output will be written to logfile, True by default',\
                                    action="store_false")
parser.add_argument('-n', '--neighs', dest = 'NNeighbors', type = int, default = config['DEFAULT']['n neighbors'],\
                                    help = 'Number of Neighbors')
parser.add_argument('-m', '--metric', dest = 'Metric', type = str, default = config['DEFAULT']['metric'],\
                    help = 'mah - mahalanobis, man - manhatan, euc - euclidean, mi3 - minkovski with p=3')

args = parser.parse_args()

Metrics = {'mah': 'mahalanobis', 'man': 'cityblock', 'euc': 'euclidean', 'mi3': 'minkowski', 'che': 'chebyshev'}

File = args.File
OutFile = args.OutFile
PointsIn = args.PointsIn
NNeighbors = args.NNeighbors
Metric = Metrics[args.Metric]

if args.ChangeCfg:
    print('Changing config')
    newConfig['DEFAULT']['file'] = File
    newConfig['DEFAULT']['outfile'] = OutFile
    newConfig['DEFAULT']['points source'] = PointsIn
    newConfig['DEFAULT']['n neighbors'] = str(NNeighbors)
    newConfig['DEFAULT']['metric'] = args.Metric


print(f'Reading... {File} {OutFile}')
product = ProductIO.readProduct(File)
width = product.getSceneRasterWidth()
height = product.getSceneRasterHeight()
name = product.getName()
description = product.getDescription()
band_names = product.getBandNames()

print("Product:     %s, %s" % (name, description))
print("Raster size: %d x %d pixels" % (width, height))
print("Start time:  " + str(product.getStartTime()))
print("Description: %s" % description)
print("End time:    " + str(product.getEndTime()))
print("Bands:       %s" % (list(band_names)))


KNNProduct = Product('KNN', 'KNN', width, height)
KNNBand = KNNProduct.addBand('KNN', ProductData.TYPE_FLOAT32)
KNNFlagsBand = KNNProduct.addBand('KNN_flags', ProductData.TYPE_UINT8)
writer = ProductIO.getProductWriter('BEAM-DIMAP')

ProductUtils.copyGeoCoding(product, KNNProduct)

KNNFlagCoding = FlagCoding('KNN_flags')
KNNFlagCoding.addFlag("1", 1, "KNN above 0")
KNNFlagCoding.addFlag("2", 2, "KNN above 1")
KNNFlagCoding.addFlag("3", 3, "KNN above 2")
KNNFlagCoding.addFlag("4", 4, "KNN above 3")
KNNFlagCoding.addFlag("5", 5, "KNN above 4")
KNNFlagCoding.addFlag("6", 6, "KNN above 5")
group = KNNProduct.getFlagCodingGroup()
group.add(KNNFlagCoding)

KNNFlagsBand.setSampleCoding(KNNFlagCoding)

KNNProduct.setProductWriter(writer)
KNNProduct.writeHeader(OutFile)


Bands = []
BandIndexes = [1, 2, 6]#[0, 1]#, 2]#, 3]#, 4]#, 5]#, 6]

for i in BandIndexes:
    Bands.append(product.getBandAt(i))
BandsNum = len(Bands)


def make_etalons(Points, Bands, Radius):
    '''Creates array of Etalons from array of Points'''
    N = len(Points)
    Etalons = list(range(N))
    for j in range(N):
        b = []
        for index, i in enumerate(Bands):
            a = np.ndarray(shape = Radius*Radius, dtype=np.float32)
            a = i.readPixels(Points[j][0], Points[j][1], Radius, Radius, a)
            b.append(np.mean(a))

        Etalons[j] = b
    return Etalons

a=-1
if PointsIn == config['DEFAULT']['points source']:#config['DEFAULT'].getboolean('ifetalons'):
    print(f"Get Etalons from config {PointsIn}, {config['DEFAULT']['etalons source']}")
    Etalons = np.array([line.rstrip('\n')[1:-3].split(',') for line in \
    open(config['DEFAULT']['etalons source'])], dtype = np.float32)
    Marks = np.array([line.rstrip('\n')[-1] for line in\
     open(config['DEFAULT']['etalons source'])], dtype = np.float32)
else:
    print(f'Creating Etalons')
    Points = [[int(line.rstrip('\n').split(' ')[0]), int(line.rstrip('\n').split(' ')[1])]\
     for line in open(PointsIn)]
    Etalons = make_etalons(Points, Bands, 2)
    Marks = [int(line.rstrip('\n').split(' ')[2]) for line in open(PointsIn)]
    print(len(Etalons), len(Points))
    with open('etalons.cfg', 'w') as f:
        for item, mark in zip(Etalons,Marks):
            f.write("%s,%s\n" % (item, mark))
    #newConfig['DEFAULT']['ifetalons'] = 'True'
if args.ChangeCfg:
    with open('KNN8.cfg', 'w') as configfile:
        newConfig.write(configfile)

t = perf_counter()

if len(Etalons) != len(Marks):
    print(f'Error Etalons don\'t fit Marks')
    sys.exit(1)


#cv = KFold(n_splits = 3, shuffle = True)
cv = StratifiedKFold(n_splits = 3)

r = np.zeros((BandsNum, width), dtype = np.float32)
#NNeighbors = 3#len(Etalons) // 4
if Metric == 'minkovski':
    clf = neighbors.KNeighborsClassifier(NNeighbors, weights='distance', algorithm = 'brute', metric = Metric, p = 3)
else:
    clf = neighbors.KNeighborsClassifier(NNeighbors, weights='distance', algorithm = 'brute', metric = Metric)

#nca = neighbors.NeighborhoodComponentsAnalysis()
#nca.fit(Etalons, Marks)
print(len(Etalons), len(Marks))
scoring = ['precision_macro', 'recall_macro', 'accuracy']
scores = cross_validate(clf, Etalons, Marks, scoring = scoring, cv=cv)
#scores2 = cross_validate(clf, nca.transform(Etalons), Marks, scoring = scoring, cv=cv)
print(scores['test_recall_macro'] , scores['test_precision_macro'], scores['test_accuracy'])
#print(scores2['test_recall_macro'] , scores2['test_precision_macro'], scores2['test_accuracy'])

#print(len(Etalons), len(nca.transform(Etalons)))




clf.fit(Etalons, Marks)
#Classes = ['nonveget', 'dark trees', 'pink grass', 'green grass',\
#           'medium trees', 'light grass', 'brown trees']
#red_index, ired_index = 2, 6

Classes = ['dark trees', 'pink grass', 'green grass',\
           'medium trees', 'light grass', 'brown trees']
Percents = np.zeros(6)#7)
#print(f'Percents is {Percents} {len(r)}')

for y in range(height):
    #print(f' Computing line {y}')
    for index, i in enumerate(Bands):
        r[index] = i.readPixels(0, y, width, 1, r[index])
    Array = list(zip(*r))
    Q = clf.predict(Array)
#    ndvi = (r[ired_index]-r[red_index])/(r[ired_index]+r[red_index])
#    ndviNormal = ndvi > 0.05
#    Q = Q * ndviNormal
    for i in range(6):#7)
        Percents[i] += np.count_nonzero(Q ==(i+1))#i)
    Flags = np.array(Q, dtype = np.int32)
    KNNFlagsBand.writePixels(0, y, width, 1, Flags)

KNNProduct.closeIO()
Percents = Percents/width/height
print(np.sum(Percents))
if np.abs(sum(Percents)-1) > 0.05:
    print('Percents Error')

if args.IfLog:
    print('Log in log.txt')
    df = DataFrame({'Classes': Classes, 'Percents': Percents})
    df.to_excel('proportion4.xlsx', sheet_name='sheet1', index=False)
    with open('log.txt', 'a') as f:
        f.write('<-------%s------->\n' % datetime.datetime.now())
        f.write('%s\n' % ' '.join(sys.argv[:]))
        f.write("Raster size: %d x %d pixels\n" % (width, height))
        f.write(f'Etalons: {len(Etalons)}, classes: {len(np.unique(Marks))}, neighbors: {NNeighbors}, metric: {Metric}\n')
        f.write(f'Class 1 %: {Percents[0]}, Class 2 %: {Percents[1]}, Class 3 %: {Percents[2]}, \n')
        f.write(f'Class 4 %: {Percents[3]}, Class 5 %: {Percents[4]}, Class 6 %: {Percents[5]}\n')
        f.write(f"scores: {scores['test_recall_macro']}, {scores['test_precision_macro']}, {scores['test_accuracy']}\n")
        f.write('Execution time is %f\n' % (perf_counter()-t))
        f.write('\n')

print('Done.')
