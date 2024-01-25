from utils.algo import Algo
from utils import tools
import numpy as np


class BS(Algo):
    """ Invests all the wealth in the best asset on the
current period  """

    PRICE_TYPE = 'raw'


    def __init__(self, b=None):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """
        super(BS, self).__init__()
        self.b = b

    def init_step(self, X):
        # print("此处是X")
        # print(X)
        # 初始化返回的数组
        period = X.shape[0]
        self.best_shock = np.zeros(X.shape[1])
        max_index = np.unravel_index(np.argmax(X.iloc[period-1]), X.shape[1])
        # print(X.iloc[period-1])
        # print(np.argmax(X.iloc[period-1]))
        # print("max_index")
        # print(max_index)
        self.best_shock[max_index] = 1
        print("best_shock")
        print(self.best_shock)
        return self.best_shock

    def step(self, x, last_b, history):
        return self.best_shock


if __name__ == '__main__':
    datasetName = 'D:\PyCharm 2019.2.6\\awhole_project\\nowPPT\data\\sp500_ratio'
    tools.quickrun(BS(), tools.dataset(datasetName))