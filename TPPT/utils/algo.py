import numpy as np
import pandas as pd
import itertools
import logging
import inspect
import copy
from utils.result import AlgoResult, ListResult
from utils import tools

try:
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


class Algo(object):
    """ Base class for algorithm calculating weights for online portfolio.
    You have to subclass either step method to calculate weights sequentially
    or weights method, which does it at once. weights method might be useful
    for better performance when using matrix calculation, but be careful about
    look-ahead bias.

    Upper case letters stand for matrix and lower case for vectors (such as
    B and b for weights).
    """

    # if true, replace missing values by last values
    REPLACE_MISSING = False

    # type of prices going into weights or step function
    #    ratio:  pt / pt-1
    #    log:    log(pt / pt-1)
    #    raw:    pt
    PRICE_TYPE = 'ratio'

    # 初始化窗口数和交易频率
    def __init__(self, min_history=None, frequency=1):
        """ Subclass to define algo specific parameters here.
        :param min_history: If not None, use initial weights for first min_window days. Use
            this if the algo needs some history for proper parameter estimation.
            如果不是“无”，则使用第一个最小窗口天数的初始权重。如果算法需要一些历史来进行正确的参数估计，请使用此选项。
        :param frequency: algorithm should trade every `frequency` periods
            算法应该每隔“频率”周期进行交易（每个交易日都交易）
        """
        self.min_history = min_history or 0
        self.frequency = frequency

    # 初始化最开始的投资比例
    # 子类重写
    def init_weights(self, m):
        """ Set initial weights.
        :param m: Number of assets.
        """
        return np.zeros(m)

    # 初始化持久变量
    def init_step(self, X):
        """ Called before step method. Use to initialize persistent variables.
                在步骤方法之前调用。用于初始化持久变量
        :param X: Entire stock returns history.
        """
        pass

    # 更新相对价格历史数据，并预测下一时期的相对价格，
    # 并根据价格给出下一时期的投资比例
    def step(self, x, last_b, history):
        """ Calculate new portfolio weights. If history parameter is omited, step
        method gets passed just parameters `x` and `last_b`. This significantly
        increases performance.
        :param x: Last returns.
        :param last_b: Last weights.
        :param history: All returns up to now. You can omit this parameter to increase
            performance.
        """
        raise NotImplementedError('Subclass must implement this!')

    # 使用历史步骤？？（不懂）
    # 这段代码是用来检查step方法是否需要使用历史数据。
    # 首先，它调用了inspect.getargspec函数，这个函数可以获取到一个函数的参数列表。
    # 然后，它检查step方法的参数列表的长度是否大于等于4。因为step方法的第四个参数就是history，所以如果step方法需要使用历史数据，那么它的参数列表的长度必须大于等于4。
    # 最后，它返回一个布尔值，表示step方法是否需要使用历史数据。
    # 这段代码的主要作用是，使得子类可以在不修改step方法的情况下，决定是否在step方法中使用历史数据。
    def _use_history_step(self):
        """ Use history parameter in step method? """
        step_args = inspect.getargspec(self.step)[0]
        return len(step_args) >= 4

    def weights(self, X, min_history=None, log_progress=True):
        """
        :param X: raw data. all data divide the first row data.？？？这里有问题，ratio数据
        :param min_history: 最小历史数据
        :param log_progress:
        :return:
        """
        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        # 返回权重。调用step方法按顺序更新公文包。将此方法子类化，风险自负。
        min_history = self.min_history if min_history is None else min_history

        # init
        # 创建一个数据集大小的矩阵，初始化为0
        B = X.copy() * 0.
        # 调用子类算法的初始化权重方法
        last_b = self.init_weights(X.shape[1])    # call class father init_weights last_b = array([0.333,0.333,...])
        # 检查初始化的权重数据类型是否为numpy.ndarry，如果是将其转化为pandas.Series
        if isinstance(last_b, np.ndarray):
            # 将numpy数组类型的last_b转换为一个pandas Series对象，
            # 其中的索引是X.columns，即X的数据列名。
            last_b = pd.Series(last_b, X.columns)

        # use history in step method?
        # 是否使用历史数据更新下一期投资比例
        use_history = self._use_history_step()

        # run algo
        self.init_step(X)
        # 对于每行数据，根据是否满足条件（如是否超过最小历史数据和是否达到交易周期），决定是否进行预测相对价格和投资比例
        # 遍历DataFrame X的每一行，返回行的索引（从0开始）和行本身（(_, x)是对应的元组，其中_是行的索引，x是行本身）
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            # 保存上一期的投资比例
            B.loc[t] = last_b

            # keep initial weights for min_history
            # 小于最小窗口数直接进入下一轮循环
            if t < min_history-1:
                continue

            # trade each `frequency` periods
            # 隔frequency时期交易一次
            if (t + 1) % self.frequency != 0:
                continue

            # predict for t+1
            if use_history:
                history = X.iloc[:t+1]
                last_b = self.step(x, last_b, history)
            else:
                last_b = self.step(x, last_b)

            # convert last_b to suitable format if needed
            # 若预测结果为numpy.maxtrix类型，转换为numpy.array类型，删除维度
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            # show progress by 10 pcts
            if log_progress:
                tools.log_progress(t, len(X), by=10)

        return B

    def _split_index(self, ix, nr_chunks, freq):
        """ Split index into chunks so that each chunk except of the last has length
        divisible by freq.
        将索引拆分为块，以便除最后一个块外的每个块的长度都可以被freq整除"""
        chunksize = int(len(ix) / freq / nr_chunks + 1) * freq
        return [ix[i*chunksize:(i+1)*chunksize] for i in range(len(ix) / chunksize + 1)]

    def run(self, S, n_jobs=1, log_progress=True):
        """ Run algorithm and get weights.
        :params S: Absolute stock prices. DataFrame with stocks in columns.
                    绝对股价。列中有库存的DataFrame
        :param show_progress: Log computation progress. Works only for algos with
            defined step method.
                    记录计算进度。仅适用于具有定义步骤方法的算法。
        :param n_jobs: run step method in parallel (step method can't depend on last weights)
                    并行运行step方法（step方法不能依赖于最后的权重）
        """
        if log_progress:
            logging.debug('Running {}...'.format(self.__class__.__name__))

        # 此处固定运行else的内容
        if isinstance(S, ListResult):
            P = S.to_dataframe()
        else:
            P = S

        self.algo_data = S.copy()

        # convert prices to proper format
        # 将数据转换为算法需要的格式
        X = self._convert_prices(P, self.PRICE_TYPE, self.REPLACE_MISSING)

        # get weights
        # 是否进行并行运算
        if n_jobs == 1:
            try:
                # 此条得到B
                B = self.weights(X, log_progress=log_progress)
            except TypeError:   # weights are missing log_progress parameter
                B = self.weights(X)

        else:
            with tools.mp_pool(n_jobs) as pool:
                ix_blocks = self._split_index(X.index, pool._processes * 2, self.frequency)
                min_histories = np.maximum(np.cumsum([0] + map(len, ix_blocks[:-1])) - 1, self.min_history)

                B_blocks = pool.map(_parallel_weights, [(self, X.loc[:ix_block[-1]], min_history, log_progress)
                                    for ix_block, min_history in zip(ix_blocks, min_histories)])

            # join weights to one dataframe
            B = pd.concat([B_blocks[i].loc[ix] for i, ix in enumerate(ix_blocks)])

        # cast to dataframe if weights return numpy array
        # 如果权重返回numpy数组，则强制转换为数据帧
        if not isinstance(B, pd.DataFrame):
            B = pd.DataFrame(B, index=P.index, columns=P.columns)

        if log_progress:
            logging.debug('{} finished successfully.'.format(self.__class__.__name__))

        # if we are aggregating strategies, combine weights from strategies
        # and use original assets
        # 如果我们正在聚合策略，请组合策略的权重并使用原始资产
        if isinstance(S, ListResult):
            B = sum(result.B.mul(B[col], axis=0) for result, col in zip(S, B.columns))
            return AlgoResult(S[0].X, B)
        else:
            # return AlgoResult(self._convert_prices(P, 'ratio'), B)
            return AlgoResult(self.algo_data, B)

    def next_weights(self, S, last_b, **kwargs):
        """ Calculate weights for next day. """
        # use history in step method?
        use_history = self._use_history_step()
        history = self._convert_prices(S, self.PRICE_TYPE, self.REPLACE_MISSING)
        x = history.iloc[-1]

        if use_history:
            b = self.step(x, last_b, history, **kwargs)
        else:
            b = self.step(x, last_b, **kwargs)
        return pd.Series(b, index=S.columns)

    def run_subsets(self, S, r, generator=False):
        """ Run algorithm on all stock subsets of length r. Note that number of such tests can be
        very large.
        :param S: stock prices
        :param r: number of stocks in a subset
        :param generator: yield results
        """
        def subset_generator():
            total_subsets = comb(S.shape[1], r)

            for i, S_sub in enumerate(tools.combinations(S, r)):
                # run algorithm on given subset
                result = self.run(S_sub, log_progress=False)
                name = ', '.join(S_sub.columns.astype(str))

                # log progress by 1 pcts
                tools.log_progress(i, total_subsets, by=1)

                yield result, name
            raise StopIteration

        if generator:
            return subset_generator()
        else:
            results = []
            names = []
            for result, name in subset_generator():
                results.append(result)
                names.append(name)
            return ListResult(results, names)


    # 转换价格格式
    @classmethod
    def _convert_prices(self, S, method, replace_missing=False):
        """ Convert prices to format suitable for weight or step function.
        Available price types are:
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == 'raw':
            # normalize prices so that they start with 1.
            r = {}
            # for name, s in S.iteritems():
            #     init_val = s.loc[s.first_valid_index()]
            #     r[name] = s / init_val
            # X = pd.DataFrame(r)
            X = S
            # 输入的是相对价值，转换为raw
            for i in range(2, len(S)):
                X.loc[i] = X.loc[i-1].multiply(S.loc[i])


            if replace_missing:
                X.loc[0] = 1.
                X = X.fillna(method='ffill')

            return X

        elif method == 'absolute':
            return S

        elif method == 'ratio':
            return S

        # elif method in ('ratio', 'log', 'ratio_1'):
        elif method in ('log', 'ratio_1'):
            # be careful about NaN values
            X = S / S.shift(1).fillna(method='ffill')
            for name, s in X.iteritems():
                X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.

            if replace_missing:
                X = X.fillna(1.)

            # return np.log(X) if method == 'log' else X
            if method == 'log':
                return np.log(X)
            elif method == 'ratio_1':
                return X - 1
            else:
                return X


        else:
            raise ValueError('invalid price conversion method')

    @classmethod
    def run_combination(cls, S, **kwargs):
        """ Get equity of algo using all combinations of parameters. All
        values in lists specified in kwargs will be optimized. Other types
        will be passed as they are to algo __init__ (like numbers, strings,
        tuples).
        Return ListResult object, which is basically a wrapper of list of AlgoResult objects.
        It is possible to pass ListResult to Algo or run_combination again
        to get AlgoResult. This is useful for chaining of Algos.

        Example:
            S = ...load data...
            list_results = Anticor.run_combination(S, alpha=[0.01, 0.1, 1.])
            result = CRP().run(list_results)

        :param S: Stock prices.
        :param kwargs: Additional arguments to algo.
        :param n_jobs: Use multiprocessing (-1 = use all cores). Use all cores by default.
        """
        if isinstance(S, ListResult):
            S = S.to_dataframe()

        n_jobs = kwargs.pop('n_jobs', -1)

        # extract simple parameters
        simple_params = {k: kwargs.pop(k) for k, v in kwargs.items()
                         if not isinstance(v, list)}

        # iterate over all combinations
        names = []
        params_to_try = []
        for seq in itertools.product(*kwargs.values()):
            params = dict(zip(kwargs.keys(), seq))

            # run algo
            all_params = dict(params.items() + simple_params.items())
            params_to_try.append(all_params)

            # create name with format param:value
            name = ','.join([str(k) + '=' + str(v) for k, v in params.items()])
            names.append(name)

        # try all combinations in parallel
        with tools.mp_pool(n_jobs) as pool:
            results = pool.map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])
        results = map(_run_algo_params, [(S, cls, all_params) for all_params in params_to_try])

        return ListResult(results, names)

    def copy(self):
        return copy.deepcopy(self)


def _parallel_weights(tuple_args):
    self, X, min_history, log_progress = tuple_args
    try:
        return self.weights(X, min_history=min_history, log_progress=log_progress)
    except TypeError:   # weights are missing log_progress parameter
        return self.weights(X, min_history=min_history)


def _run_algo_params(tuple_args):
    S, cls, params = tuple_args
    logging.debug('Run combination of parameters: {}'.format(params))
    return cls(**params).run(S)

