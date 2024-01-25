import os
import csv
import datetime
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import heapq
class TopLowStocksSelectors:
    def __init__(self, b_expectedReturn, dataset_nStocks, nTopStocks, nLowStocks, loopTrainEpochs=20, batchsize=30):
        """

        :param b_expectedReturn:  is the expected return, it is a real number.
        :param dataset_nStocks:  total number of  stocks, is also the dimension of price data.
        :param nTopStocks: the number of top weight
        :param nLowStocks: the number of low weight
        :param loopTrainEpochs:   the number of scans of the datasets.
        :param batchsize:  size of a batch
        """
        self.b_expectedReturn = b_expectedReturn
        self.dataset_nStocks = dataset_nStocks
        self.nTopStocks = nTopStocks
        self.nLowStocks = nLowStocks
        self.loopTrainEpochs = loopTrainEpochs
        self.batchsize = batchsize

        # the file to save training result:
        # self.savefile = '/media/aze/' + str(datetime.datetime.now()) + '.csv'
        # self.file = open(self.savefile, 'w')
        # self.csv_writer = csv.writer(self.file)
        #
        # # the file to save final result
        # self.resfile =  '/media/aze/' + str(datetime.datetime.now()) + '.csv'
        # self.refile = open(self.resfile, 'w')
        # self.csv_writer2 = csv.writer(self.refile)

        # random weight
        self.b = torch.rand(self.dataset_nStocks, 1).double()
        # self.b = torch.ones(self.dataset_nStocks, 1).double() / self.dataset_nStocks
        self.b.requires_grad = True
        self.optimizer = optim.SGD([self.b], lr=1e-8, momentum=0.9)

        self.rList = list(torch.arange(0, 1, 0.01))
        self.rList.sort()
        self.alpha_r = 0.0
        for r in self.rList:
            self.alpha_r += 1.0 / ((r + 1.1) ** 4)

    def initBalance(self, dfHistory):
        ndays = dfHistory.shape[0] - 1
        nstocks = dfHistory.shape[1]
        last_price = dfHistory.iloc[ndays]
        last_price = np.array(last_price)
        last_price = last_price/sum(last_price)
        last_price = last_price.reshape(nstocks, 1)
        init_b = torch.tensor(last_price, dtype=None, device=None, requires_grad=True).double()
        return init_b


    def  getTopLowStocks(self, dfHistory):
        """

        :param dfHistory: the price data.
        :return:  return type SERIES.[topStocks_weights,  lowStocks_weights], whose sum is 1.
        """

        balance = self._trainData(dfHistory)
        balance_list = []
        for i in balance:
            balance_list.append(i[0])
        balance_series = pd.Series(balance_list, index=dfHistory.columns)

        return balance_series

    def _trainData(self, dfHistory):
        """

        :param dfHistory:  price data of all of stocks.
        :return: return average weight.
        """



        self._createTrainLoader(dfHistory)
        # device = torch.device("cuda:0" if torch.cuda.is_availabl e() else "cpu")
        # self.csv_writer.writerow(["epoch", "day", "index of top", "weights of top", "index of low",  "weights of low", "loss"])
        self.b = torch.rand(self.dataset_nStocks, 1).double()
        # print("##self.b##", self.b)
        # self.b = self.initBalance(dfHistory)
        # self.b = torch.ones(self.dataset_nStocks, 1).double() / self.dataset_nStocks
        self.b.requires_grad = True
        self.optimizer = optim.SGD([self.b], lr=1e-8, momentum=0.9)
        # self.optimizer = optim.adam([self.b], lr=1e-12, momentum=0.9) #dzw

        for epoch in range(self.loopTrainEpochs):

            self.optimizer.zero_grad()
            loss = self._loss(self.b, self.trainloader)
            # loss = self._loss(self.b, self.trainloader).cuda(device)
            loss.backward()
            self.optimizer.step()

            # Normalized and calculate top stocks' index and value
            # b_normal = torch.softmax(self.b, dim=0)
            # b_list = list(b_normal)
            # top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
            # last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
            # top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
            # last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

            # self.csv_writer.writerow(
            #     [
            #         # build a list of
            #         epoch,
            #         [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
            #         top_index, list(top_value.detach().numpy()),
            #         last_index, list(last_value.detach().numpy()),
            #         loss.cpu().detach().numpy()
            #     ])

            # if epoch % 5 == 4:
            #     lr = self.optimizer.param_groups[0]['lr']
            #     # print('epoch', epoch, 'learing rate: ....', lr)
            #     self.optimizer.param_groups[0]['lr'] = lr / 2

        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal.detach().numpy())
        # top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
        # last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
        # last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

        # self.csv_writer2.writerow(
        #     ["day", "index of top", "weights of top", "index of low", "weights of low"])
        # self.csv_writer2.writerow([[str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
        #                            top_index, list(top_value.detach().numpy()),
        #                            last_index, list(last_value.detach().numpy())])




        top_value = list(top_value.detach().numpy())

        # self.outputTopLowData_csv(dfHistory, top_index, last_index)

        # y = np.zeros((len(self.b), 1))
        # j = 0
        # last_value_sum = np.sum(top_value)
        # for i in top_index:  # like [2, 5]
        #     y[i] = top_value[j] / last_value_sum
        #     j = j + 1
        #
        # # b_normal = torch.softmax(self.b, dim=0)
        # y = b_normal.detach().numpy()

        return b_list



    def outputTopLowData_csv(self, dfHistory, top_index, last_index):
        index = top_index + last_index
        path = os.getcwd() + '/topLowStocksData/' + str(datetime.datetime.now()) + '.csv'

        dfData = dfHistory.iloc[:, index]
        dfData.to_csv(path)



    def _loss(self, x, loader):
        '''
            calculate the average loss between 0 and current batch
        :param x:  weight, self.b

        :param loader:  loader for the dataset from which we compute the loss.
        :return:  loss a number
        '''

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        y = torch.softmax(x, dim=0)
        dotProds = torch.tensor(0.0)
        reguItem = 0.0
        for j, data in enumerate(loader):
            # if j != len(loader) - 1:
            #     continue
            ksaiBatch = data  # get one batch
            # ksaiBatch = ksaiBatch
            Ex = torch.matmul(ksaiBatch, y)
            # Ex:  (batchSize, 1)
            dotProds = dotProds + torch.sum(Ex - 1)  # sum over each samples in each batch

            for r in self.rList:
                # reguItem += (1 / self.batchSize * torch.sum(torch.exp(b - Ex - r)) - alpha_r)
                # reguItem += torch.sum(torch.exp(self.b_expectedReturn - Ex - r))
                reguItem += torch.sum(torch.exp(self.b_expectedReturn - (Ex - 1) - r))  #dzw
                # torch.sum over batchSize rows of each batch



        # sum1 = dotProds / ((ibatch + 1) * self.batchsize)
        datasize = len(loader) * self.batchsize
        sum1 = dotProds/datasize

        sum2 = reguItem / datasize
        sum3 = 1e+08 * (sum2 - self.alpha_r)

        loss = -sum1 + sum3
        return loss


    def _createTrainLoader(self, dfHistory):
        """

        :param dfHistory: the price data
        :return: return trainLoader
        """

        history_numpy = np.array(dfHistory)
        history_tensor = torch.from_numpy(history_numpy)
        loader = torch.utils.data.DataLoader(history_tensor, batch_size=self.batchsize,
                                             shuffle=True, num_workers=2)
        self.trainloader = loader
