# from ..algo import Algo
from utils.algo import Algo
from utils.result import ListResult
import numpy as np
import pandas as pd
# from .. import tools
from utils import tools
# from MyLogger import MyLogger
import os
import datetime

class OLMAR(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

        # self.logger = MyLogger('olmar_log')
        self.histLen = 0  # yjf.



    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        self.histLen = history.shape[0]
        # print("%%%%%%last_b", last_b, x)
        x_pred = self.predict(x, history.iloc[-self.window:])

        b = self.update(last_b, x_pred, self.eps)

        # print(len(history), b)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        return (history / x).mean()


    def update(self, b, x, eps):
        """

        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 10
        :return:  weight
        """

        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)

        # print('b: ', b)
        # print('x: ', x)
        b_dot_x = np.dot(b, x)
        # print('b_dot_x: ', b_dot_x)
        gap = (eps - np.dot(b, x))
        # print('gap: ', gap)
        x_avg_norm = np.linalg.norm(x - x_mean)**2
        # print('x_avg_norm: ', x_avg_norm)

        gap_n = gap / x_avg_norm
        # print('gap_n: ', gap_n)


        # lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)
        lam = max(0.0, gap_n)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        b = b + lam * (x - x_mean)
        bn = tools.simplex_proj(b)

        return bn




if __name__ == '__main__':
    datasetName = 'D:\PyCharm 2019.2.6\\awhole_project\\nowPPT\data\\nyse_o_ratio'
    tools.quickrun(OLMAR(), tools.dataset(datasetName))
