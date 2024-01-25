import logging

from utils.result import ListResult
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class MultiShower:

    def __init__(self, fileName):
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # dtMark = str(datetime.datetime.now()) + '_'
        self.fileName = 'E:\PyCharm 2022.1\\awhole_project\\nowPPT\pic\\' + fileName +'.eps'
        # print("self.fileName:",self.fileName)

    def show(self, resultList, algoNameList, yLable='Total Wealth', logy1=True):

        res = ListResult(resultList,
                         algoNameList)
        d = res.to_dataframe()
        portfolio = d.copy()
        print("portfolio:",portfolio)
        colorDic = {'TPPT': 'C0', 'BAH': 'C1', 'CORN': 'C2',
                    'ONS': 'C4', 'RMR': 'C5', 'UP': 'C9', "CRP":'C3','EMA':'C6','PPT':'C7'}

        color = ["Blues", "Reds", "Blues", "Reds", "Blues", "Reds", "Blues", "Reds", "Blues"]
        # color = np.array(color)
        color = pd.DataFrame(color)
        plt.gcf().set_size_inches(14, 8)
        plt.margins(x=0)

        for columnName in portfolio:
            ax = portfolio[columnName].plot(linewidth=1., color=colorDic[columnName],
                                            label=columnName)
        ax.set_ylabel(yLable)
        ax.set_xlabel('day')
        plt.legend(frameon=True)
        plt.grid(True)
        plt.show()

        # fig1 = plt.gcf()
        # plt.show()
        # fig1.savefig(self.fileName, format='eps')


