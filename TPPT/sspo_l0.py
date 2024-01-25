import json
import math
import os

import pandas as pd

from utils.algo import Algo
from utils import tools
import numpy as np


class SSPO_L0(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever.  """

    PRICE_TYPE = 'raw'

    # REPLACE_MISSING = True

    def __init__(self, K, C, lamda, gamma, rho, window=5):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """

        self.window = window
        self.K = K
        self.C = C
        self.lamda = lamda
        self.gamma = gamma
        self.rho = rho
        self.histLen = 0  # yjf.
        # self.data_path = "C:\\Users\\zay\\Desktop\\portfolio\\universal\\data\\" + d + ".pkl"
        # self.all_data = pd.read_pickle(self.data_path)
        super(SSPO_L0, self).__init__(min_history=self.window)

    def init_weights(self, m):
        return np.ones(m) / m

    def calDistance(self, vector1, vector2):
        if len(vector1) != len(vector2):
            return -1
        distance = 0
        for i in range(len(vector1)):
            distance += (vector1[i] - vector2[i]) ** 2
        return math.sqrt(distance)

    def sign(self, x):
        if x > 0:
            res = 1
        elif x < 0:
            res = -1
        else:
            res = 0
        return res

    def positive_element(self, x):
        # print(x)
        for i in range(len(x)):
            # print(x[i])
            if x[i] <= 0:
                x[i] = 0
        return x

    def admm(self, m, w_t, fai_t, K, C, lamda, gamma, rho):
        w_t = np.array(w_t)
        w_k = np.zeros(m)
        z = np.zeros(m)
        g = np.zeros(m)
        beta = np.zeros(m)
        w_k_old = np.ones(m)
        z_old = np.ones(m)
        g_old = np.ones(m)
        beta_old = np.ones(m)
        # rho_list = [rho for i in range(m)]
        one_vector = np.ones(m)
        identity_matrix = np.eye(m)
        ones_matrix = np.ones((m, m))
        k = 0
        while k < 10 or self.calDistance(w_k, w_k_old) > 5*10 ** (-7):
            # while k < 10 or (self.calDistance(w_k, w_k_old) > 10 ** (-20) and \
            #                  self.calDistance(z, z_old) > 10 ** (-20) and \
            #                  self.calDistance(g, g_old) > 10 ** (-20) and \
            #                  self.calDistance(beta, beta_old) > 10 ** (-20)):
            # while self.calDistance(w_k, w_k_old) > 0.0000000000001:
            # print("z", self.calDistance(z, z_old), list(z), list(z_old))
            # print("g", self.calDistance(g, g_old), list(g), list(g_old))
            # print("w", self.calDistance(w_k, w_k_old), list(w_k), list(w_k_old))
            # print("beta", self.calDistance(beta, beta_old), list(beta), list(beta_old))
            w_k_old = np.array(w_k)
            z_old = np.array(z)
            g_old = np.array(g)
            beta_old = np.array(beta)

            # z-update
            delta = w_k + beta / rho
            # print(w_k,beta,rho)
            delta_sort = sorted(delta, key=abs)
            for i in range(m):
                if abs(delta[i]) >= abs(delta_sort[m - K]):
                    z[i] = delta[i]
                else:
                    z[i] = 0

            # g-update
            v_add = self.positive_element(abs(w_k - w_t) - gamma * one_vector)
            for i in range(m):
                g[i] = self.sign(w_k[i] - w_t[i]) * (v_add[i])

            # w_k-update

            A = ((rho + lamda / gamma) * identity_matrix + C * ones_matrix)
            w_k = np.dot(np.linalg.inv(A), (
                    lamda / gamma * (g + w_t) + rho * z + C * one_vector - beta - fai_t))
            # print("wk",w_k)
            w_k = self.positive_element(w_k)

            # beta-update
            beta = beta + rho * (w_k - z)

            k = k + 1
        print(k)

        return list(w_k)

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        self.histLen = history.shape[0]
        relative_p = self.predict(x, history.iloc[-self.window:])
        fai_t = [0 for i in range(history.shape[1])]
        for i in range(history.shape[1]):
            fai_t[i] = -1.1 * math.log(relative_p[i], 2) - 1
        # print(fai_t)
        b = self.admm(history.shape[1], last_b, fai_t, self.K, self.C, self.lamda, self.gamma, self.rho)
        b = tools.simplex_proj(b)
        # print(b)

        return b

    def predict(self, x, history):
        """ Predict returns on next day. """
        result = []
        for i in range(history.shape[1]):
            temp = max(history.iloc[:, i]) / x[i]
            result.append(temp)
        return result

if __name__ == '__main__':
    datasetName ="D:\SZU_homework\在线投资组合\other_algo\data\\nyse_n" \
                 "_ratio"
    tools.quickrun(SSPO_L0(K=3, C=10, lamda=0.1, gamma=1, rho=10), tools.dataset(datasetName))