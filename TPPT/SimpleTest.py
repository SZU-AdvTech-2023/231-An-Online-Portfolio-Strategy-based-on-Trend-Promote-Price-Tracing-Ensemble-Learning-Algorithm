import csv
import os
import pandas as pd
import heapq
import matplotlib.pyplot as plt
import algos
import pickle
from utils.algo import Algo
from utils import tools
import random, datetime
import logging
from MyLogger import MyLogger

import numpy as np

# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import matplotlib


# print('type: ', type(matplotlib.rcParams['savefig.dpi']), 'va: ', matplotlib.rcParams['savefig.dpi'])

from MultiShower1 import MultiShower
# from SimpleSaver import SimpleSaver

# increase the size of graphs
# matplotlib.rcParams['savefig.dpi'] *= '1.5'

class Tester:

    def __init__(self):
        self.data = None
        self.algo = None
        self.result = None
        self.X = None
        # self.logger = MyLogger('PTester_summary')
        # self.saver = SimpleSaver()
        self.datasetName = None
        self.NStocks = 0

    def createDataSet(self, datasetName):
        # load data using tools module
        s = "E:\PyCharm 2022.1\\awhole_project\\nowPPt\\data\\"+ datasetName +"_ratio"
        self.data = tools.dataset(s)
        # print("self.data:",self.data)
        self.datasetName = datasetName
        self.NStocks = self.data.shape[1]
    # 创造算法
    def createAlgo(self):

        # set algo parameters
        dataset_nStocks = self.data.shape[1]
        # self.algo = algos.PPT()
        # self.algo = algos.BAH()
        # self.algo = algos.BS()
        # self.algo = algos.RMR()
        # self.algo = algos.Anticor()
        # self.algo = algos.CORN()
        # self.algo = algos.OLMAR()
        self.algo = algos.TPPT()
        # self.algo = algos.ONS()
        # self.algo = algos.UP()
        # self.algo = algos.CRP()
        # self.algo = algos.EMA()

        return self.algo
    # 运行算法
    def runAlgo(self):
        data = self.data
        print("data:",data)
        self.result = self.algo.run(self.data)
        print("self.result.search():",self.result.search())

    # 得到算法结果对象的名字
    def getDataSetNameWithDT(self):
        # tMark = str(datetime.datetime.now())
        return self.datasetName + '_' + str(self.NStocks) + '_'
    # 保存结果
    def saveresult(self):
        path = './resultSave/' + self.getDataSetNameWithDT()
        # self.result.save(path + 'PPT')
        # self.result.save(path + 'BAH')
        # self.result.save(path + 'BS')
        # self.result.save(path + 'RMR')
        # self.result.save(path + 'Anticor')
        # self.result.save(path + 'CORN')
        # self.result.save(path+'OLMAR')
        self.result.save(path+'TPPT_min')
        # self.result.save(path + 'ONS')
        # self.result.save(path + 'UP')
        # self.result.save(path + 'CRP')
        # self.result.save(path + 'EMA')

    # 展示结果
    def showResult(self):
        # 加载数据展示结果
        path = './resultSave/' + self.getDataSetNameWithDT()
        with open(path+'BAH', 'rb') as f:
            result_bah =  pickle.load(f)
        f.close()
        with open(path +'BS', 'rb') as f:
            result_bs = pickle.load(f)
        f.close()
        with open(path+'CORN', 'rb') as f:
            result_corn =  pickle.load(f)
        f.close()
        with open(path+'Anticor', 'rb') as f:
            result_anticor =  pickle.load(f)
        f.close()
        with open(path+'OLMAR', 'rb') as f:
            result_olmar =  pickle.load(f)
        f.close()
        with open(path+'RMR', 'rb') as f:
            result_rmr =  pickle.load(f)
        f.close()
        with open(path+'PPT', 'rb') as f:
            self.result =  pickle.load(f)
        f.close()
        with open(path+'TPPT', 'rb') as f:
            result_tppt =  pickle.load(f)
        f.close()
        with open(path+'ONS', 'rb') as f:
            result_ons =  pickle.load(f)
        f.close()
        with open(path+'UP', 'rb') as f:
            result_up =  pickle.load(f)
        f.close()
        with open(path+'CRP', 'rb') as f:
            result_crp =  pickle.load(f)
        f.close()
        with open(path+'EMA', 'rb') as f:
            result_ema =  pickle.load(f)
        f.close()
        # 画图CW
        portfolio_label = 'TPPT'
        from algos.anticor import Anticor
        from algos.bah import BAH
        from  algos.bs import BS
        from  algos.corn import CORN
        from algos.olmar import OLMAR
        from  algos.rmr import RMR
        from algos.ppt import PPT
        #
        # result_anticor = Anticor().run(self.data)
        # result_bah = BAH().run(self.data)
        # result_bs = BS().run(self.data)
        # result_corn = CORN().run(self.data)
        # result_olmar = OLMAR().run(self.data)
        # result_rmr = RMR().run(self.data)
        #
        # result_anticor.save(path+'ANTICOR')
        # result_bah.save(path + "BAH")
        # result_bs.save(path+'BS')
        # # result_corn.save(path+'CORN')
        # result_olmar.save(path+'OLMAR')
        # result_rmr.save(path+'RMR')


        ms = MultiShower(self.getDataSetNameWithDT() + '_Result_')
        # ms.show([self.result],
        #         ['PPT'],
        #         yLable=self.datasetName + ' Total Wealth=')
        # ms.show([self.result,result_bah,result_bs,result_rmr,result_olmar,result_anticor,result_corn,result_tppt],
        #         ['PPT','BAH','BestStock','RMR','OLMAR','ANTICOR','CORN','TPPT'],
        #         yLable=self.datasetName + ' Total Wealth=')
        ms.show([result_bah,result_rmr,result_corn,result_tppt,result_ons,result_up,result_crp,result_ema,self.result],
                ['BAH','RMR','CORN','TPPT','ONS','UP','CRP','EMA','PPT'],
                yLable=self.datasetName + ' Total Wealth=')
        # ms.show([result_olmar_rss_bah], ['OLMAR_S1', 'PPT', 'PASS'], yLable=self.datasetName + ' Total Wealth=' + str(fee))

        # plt.show()

    @staticmethod
    def testSimple():
        # datasets = ["djia","sp500","tse","nyse_n","nyse_o"]
        # datasets = ["sp500"]
        datasets = ['djia']
        for d in datasets:
            print("datasetName:",d)
            print("Algo:TPPT")
            t = Tester()
            t.createDataSet(d)
            t.createAlgo()
            # t.runAlgo()
            # t.saveresult()
            t.showResult()

if __name__ == '__main__':
    Tester.testSimple()