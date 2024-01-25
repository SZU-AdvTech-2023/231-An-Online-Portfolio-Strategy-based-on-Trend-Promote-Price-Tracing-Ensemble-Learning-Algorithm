import heapq
import pandas as pd
import numpy as np

from .TopLowStocksSelectors_old import TopLowStocksSelectors
from .olmar import OLMAR
import random
import csv


class GetRemainder():
    """
    cut dataset according percentage or dataset columns
    """
    def __init__(self, history, dataset=None, percentage=None, index = None, selectedIndex = None,ndays=None, expectationReturn=None):
        """

        :param history:
        :param dataset:
        :param percentage:
        :param index: 1 drop top stock, 0 draw low stock
        :param selectedIndex: selected asset's index
        """
        self.history = history
        self.dataset = dataset
        self.percentage = percentage
        self.filepath = "/home/aze/project/UPalgoTest/universal/data" + "/" + dataset + ".pkl"
        self.nStocks = history.shape[1]
        self.index = index
        self.selectedIndex = selectedIndex

        # self.filepath1 = "/home/aze/project/UPalgoTest/universal/data" + "/" + dataset + "_ratio.pkl"
        # self.history = pd.read_pickle(self.filepath1)[:history.shape[0]]
        self.ndays = ndays
        self.expectationReturn = expectationReturn

        # self.savePath = "/home/aze/project/UPalgoTest/dataSetSave/" + "djia_hole" + ".csv"
        # self.f = open(self.savePath, 'w', newline='')
        # self.csv_writer1 = csv.writer(self.f)
        #
        # self.savePath = "/home/aze/project/UPalgoTest/dataSetSave/" + "djia_subset" + ".csv"
        # self.f = open(self.savePath, 'w', newline='')
        # self.csv_writer2 = csv.writer(self.f)

    def calNumstocks(self, percentage=None, filepath=None):
        """calculate how many stocks selected"""

        # if dataset == None:
        #     dataset = self.dataset
        if percentage == None:
            percentage = self.percentage

        if filepath == None:
            filepath = self.filepath

        df = readPKL(filepath)
        numstocks = round(df.shape[1] * percentage)
        return numstocks

        # if isinstance(percantage, list):
        #     numstocks = []
        #     df = readPKL(filepath)
        #     for i in percantage:
        #         numstocks.append(round(df.shape[1] * percantage))
        #     return numstocks

    def getIndex(self, b=None, numstocks=None):
        """get the index of part of stocks that the weight in rss is large"""

        if b == None:
            dataset = pd.read_pickle(self.filepath)
            nstocks = dataset.shape[1]
            # expectationReturn = self.expectationReturn*self.history.shape[0]
            expectationReturn = 0.0045*self.history.shape[0]
            # history = dataset.iloc[:self.history.shape[0]]
            RSS = TopLowStocksSelectors(expectationReturn, nstocks, 3, 3, batchsize=30)
            # print("expectation:", expectationReturn)
            # history = self.history.iloc[-90:]
            # b = RSS.getTopLowStocks(self.history.iloc[-self.ndays:])
            b = RSS.getTopLowStocks(self.history)

            self.whole_b = b
            # self.csv_writer1.writerow(b)

            # print("b:", b)
            # if self.history.shape[0] < 180 :
            #     b = RSS.getTopLowStocks(self.history)
            # else:
            #     b = RSS.getTopLowStocks(self.history.iloc[-180:])
            b = list(b)
        if numstocks == None:
            numstocks = self.calNumstocks()
        # if isinstance(numstocks, int):
        dataset = pd.read_pickle(self.filepath)
        lowStockIndex = map(b.index, heapq.nsmallest(numstocks, b))
        lowStockIndex = list(lowStockIndex)
        topStockIndex = map(b.index, heapq.nlargest(dataset.shape[1]-numstocks, b))
        # topStockIndex = map(b.index, heapq.nlargest(3, b))
        topStockIndex = list(topStockIndex)
        if self.index == 0 :
            self.b_subset = heapq.nlargest(dataset.shape[1]-numstocks, b)
            # self.csv_writer2.writerow(b_subset)
        else:
            self.b_subset = heapq.nsmallest(numstocks, b)
            # self.csv_writer2.writerow(b_subset)
        # print("topStockIndex:", topStockIndex)
        # print("b:", b)

        # get lowStockIndex dzw
        # allIndex = [i for i in range(dataset.shape[1])]
        # for i in topStockIndex:
        #     allIndex.remove(i)
        # lowStockIndex = allIndex


        # print("lowStockIndex:", lowStockIndex)
        # print("topStockIndex:", topStockIndex)
        return lowStockIndex, topStockIndex

        # stockIndexLists = []
        # for i in numstocks:
        #     stockIndexLists = heapq.nlargest(b, numstocks)
        #     stockIndexLists.append(stockIndexLists)
        #
        # return stockIndexLists

    def cutDataset(self, ndays=None, filepath=None):
        """cut history according to columns"""
        if filepath == None:
            filepath = self.filepath
        stockIndex = self.getIndex()[self.index]
        #save selected sub asste
        selectedIndex = self.getIndex()[self.selectedIndex]
        

        df = self.history.copy()
        # df = readPKL(self.filepath)

        itemLists = []
        count = 0
        for item in df:
            if count in stockIndex:
                itemLists.append(item)
            count += 1
        df = df.drop(itemLists, axis=1)
        df = df.iloc[:ndays]
        return df, selectedIndex, self.whole_b, self.b_subset

    def randomIndex(self):
        stockLists = [i for i in range(self.nStocks)]
        numstocks = self.calNumstocks()
        lowStockIndex = random.sample(stockLists, numstocks)
        topStockIndex = []
        for item in stockLists:
            if item not in topStockIndex:
                topStockIndex.append(item)
        return lowStockIndex, topStockIndex

def savedAsPKL(df, filepath):
    """save file as pickle"""
    pd.DataFrame.to_pickle(df, filepath)
    return pd


def readPKL(filepath):
    """read pickle"""
    df = pd.read_pickle(filepath)
    return df


def olmarBalance(history):
    """calculate last balance of olmar"""
    olmar = OLMAR()
    m = history.shape[1]
    b = olmar.init_weights(m)
    if history.shape[0] > 5:
        for i in range(6, history.shape[0]):
            b = olmar.step(history.iloc[i-1], b, history.iloc[:i])
    return b


def getContinuousPart(b, percentage1, percentage2):
    """get continuous part index of balance
     param b:balance of rss. type is list
     param percentage1: top stock percentage
     param percentage2: low stock percentage"""
    topStocks = round(len(b) * percentage1)
    lowStocks = round(len(b) * percentage2)
    lowStockIndex = map(b.index, heapq.nsmallest(lowStocks, b))
    lowStockIndex = list(lowStockIndex)
    topStockIndex = map(b.index, heapq.nlargest(topStocks, b))
    topStockIndex = list(topStockIndex)
    allIndex = []
    # twoSideStockIndex = lowStockIndex + topStockIndex
    for i in range(len(b)):
        allIndex.append(i)
    for j in range(len(topStockIndex)):
        allIndex.remove(topStockIndex[j])
    for h in range(len(lowStockIndex)):
        allIndex.remove(lowStockIndex[h])

    return allIndex

class SortReturn():
    def __init__(self, x, history, percentage, Nstocks, index=None):
        """sort stock according to total return,
        :param x :last price from raw dataset meaning total return
        """
        self.total_wealth = x
        self.history = history
        self.percentage = percentage
        # self.filepath = "/home/aze/project/UPalgoTest/universal/data" + "/" + datasetName + ".pkl"
        self.Nstocks = Nstocks
        self.index = index

    def calNumStocks(self):
        Nstocks = round(self.Nstocks * self.percentage)
        return Nstocks

    def getIndex(self):
        total_wealth = list(self.total_wealth)
        numLowStocks = self.calNumStocks()
        lowStockIndex = map(total_wealth.index, heapq.nsmallest(numLowStocks, total_wealth))
        lowStockIndex = list(lowStockIndex)
        topStockIndex = map(total_wealth.index, heapq.nlargest(self.Nstocks - numLowStocks, total_wealth))
        # topStockIndex = map(b.index, heapq.nlargest(3, b))
        topStockIndex = list(topStockIndex)
        randomIndex = random.sample(range(self.Nstocks), numLowStocks)
        return lowStockIndex, topStockIndex, randomIndex

    def cutDataset(self, ndays=None, filepath=None):
        """cut history according to columns"""
        stockIndex = self.getIndex()[self.index]
        df = self.history.copy()
        # df = readPKL(self.filepath)

        itemLists = []
        count = 0
        for item in df:
            if count in stockIndex:
                itemLists.append(item)
            count += 1
        df = df.drop(itemLists, axis=1)
        df = df.iloc[:ndays]
        return df