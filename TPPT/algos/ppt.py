from utils.algo import Algo
from utils import tools
import numpy as np

import matplotlib.pyplot as plt

class PPT(Algo):
    """ 追踪增长趋势，通过预测的下一期相对价格，策划出下一期的投资比例  """

    PRICE_TYPE = 'raw'

    # 数据集没有缺失值，不需要replace
    # REPLACE_MISSING = True

    def __init__(self, window=5, eps=100):
        """
        :params window: 窗口数量.
        :param eps : 超参，默认100效果最好
        """

        self.window = window
        self.eps = eps
        self.histLen = 0

        super(PPT, self).__init__(min_history=self.window)

    # 初始化投资比例
    # m : 股票数量
    def init_weights(self, m):
        return np.ones(m) / m

    # 更新相对价格历史数据，并预测下一时期的相对价格，
    # 并根据价格给出下一时期的投资比例
    def step(self, x, last_b, history):
        """
        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        # 数据集的行数，表示时期
        self.histLen = history.shape[0]

        # 调用 predict 方法来预测未来的股票价格走势。
        # x ：当前的股票价格信息，
        # history.iloc[-self.window:] ：最近的历史价格信息，表示获取最后五个时期的相对价格
        # self.window ：用于预测的时间窗口长度
        x_pred = self.predict(x, history.iloc[-self.window:])
        # print(x_pred)
        # last_b ： 上一次的投资比例
        # x_pred ： 预测的下一期相对价格
        b = self.update(last_b, x_pred, self.eps)

        # print(b)
        # print(self.histLen, len(b), list(b))

        return b

    # 返回下一时期每只股票预测的相对价格
    def predict(self, x, history):
        """ Predict returns on next day. """
        result = []
        # 遍历每一支股票的窗口并预测
        for i in range(history.shape[1]):
            # 从窗口的数据中找到最大的相对价格，除以当前的相对价格
            temp = max(history.iloc[:, i]) / x[i]
            result.append(temp)
        return result

    # 根据预测的相对价格，返回下一时期的投资比例
    def update(self, b, x, eps):
        """
        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 100
        :return:  weight
        """
        # 创建单位矩阵，再将每个元素减去 1 / len(b)
        # 确保新的投资组合比例之和为1
        identity_matrix = np.eye(len(b)) - 1 / len(b)
        # 存储新的投资比例
        x_hat = []

        # 计算投资组合比例的总和
        count_x_hat = 0

        # 计算有帽子的x
        # 遍历每只股票
        for i in range(len(b)):
            # 通过点乘计算新的投资比例，identity_matrix[i]表示单位矩阵第i行
            # print(identity_matrix[i], "+", x)
            temp = np.dot(identity_matrix[i], x)
            # print(type(temp))
            x_hat.append(temp)
            # print(np.around(np.dot(identity_matrix[i], x),3))
            count_x_hat = count_x_hat + abs(temp)
        # print("x_hat")
        # print(x_hat)

        # 计算新的投资组合的范数，即他们的绝对值之和
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
    # datasetName = "D:\PyCharm 2019.2.6\\awhole_project\\nowPPT\\data\\sp500_ratio"
    datasetName = "E:\PyCharm 2022.1\\awhole_project\\nowPPt\\data\\sp500_ratio"
    tools.quickrun(PPT(), tools.dataset(datasetName))
    # list = [5,10,50,100,500,1000]
    # list_string = ['5','10','50','100','500','1000']
    # listresult =[]
    # for i in list:
    #     print("eps=",i)
    #     listresult.append(tools.quickrun(PPT(eps=i), tools.dataset(datasetName)))
    # print("listresult:",listresult)
    # plt.bar(list_string,listresult,width=0.2)
    # plt.title('CWs of PPT with respect to eps on nyse_n')
    # plt.show()
