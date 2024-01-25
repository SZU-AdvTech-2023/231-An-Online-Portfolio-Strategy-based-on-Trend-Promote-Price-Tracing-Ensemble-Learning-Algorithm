import json
import os

import pandas as pd

from utils.algo import Algo
from utils import tools
import numpy as np


class EMA(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever.  """

    PRICE_TYPE = 'raw'

    # REPLACE_MISSING = True

    def __init__(self, window=5, eps=100):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """

        self.window = window
        self.eps = eps
        self.histLen = 0
        # self.all_data = pd.read_pickle(self.data_path)
        super(EMA, self).__init__(min_history=self.window)

    def init_weights(self, m):
        return np.ones(m) / m

    def cal_slope(self, t1, t2, history):
        # p_t = self.all_data.iloc[self.histLen-1]
        # p_t_k = self.all_data.iloc[self.histLen-1-k]
        p_t = history.iloc[-t1, :]
        p_t_k = history.iloc[-t2, :]
        res = (p_t - p_t_k) / (t2 - t1)
        # print(res)
        return res

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        self.histLen = history.shape[0]
        k = [0 for i in range(history.shape[1])]
        for t1 in range(1, 5):
            for t2 in range(t1 + 1, 6):
                k = k + self.cal_slope(t1, t2, history.iloc[-self.window:])
        # print(k)
        x_pred = self.predict(x, history.iloc[-self.window:], k, 0.5)
        b = self.update(last_b, x_pred, self.eps)

        # print(self.histLen, len(b), list(b))

        return b

    def predict(self, x, history, k, afa):
        """ Predict relative price on next day. """
        result = []
        for i in range(history.shape[1]):
            p_hat = afa * ((1 - afa) ** 4) * history.iloc[-5, i] + afa * ((1 - afa) ** 3) * history.iloc[
                -4, i] + afa * ((1 - afa) ** 2) * history.iloc[-3, i] + afa * (1 - afa) * history.iloc[
                        -2, i] + afa * history.iloc[-1, i]
            result.append(p_hat / x[i])
        return result

    def update(self, b, x, eps):
        """

        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 100
        :return:  weight
        """

        identity_matrix = np.eye(len(b)) - 1 / len(b)
        x_hat = []

        count_x_hat = 0

        for i in range(len(b)):
            temp = np.dot(identity_matrix[i], x)
            # print(type(temp))
            x_hat.append(temp)
            # print(np.around(np.dot(identity_matrix[i], x),3))
            count_x_hat = count_x_hat + abs(temp)
        # print(x_hat)

        x_hat_norm = np.linalg.norm(x_hat)
        # update portfolio
        for i in range(len(x_hat)):
            x_hat[i] = x_hat[i] * eps / x_hat_norm

        if count_x_hat == 0:
            b = b
        else:
            for i in range(len(x_hat)):
                b[i] = b[i] + x_hat[i]

        # project it onto simplex
        bn = tools.simplex_proj(b)

        return bn

if __name__ == '__main__':
    datasetName = "E:\PyCharm 2022.1\\awhole_project\\nowPPT\data\\djia_ratio"
    tools.quickrun(EMA(), tools.dataset(datasetName))