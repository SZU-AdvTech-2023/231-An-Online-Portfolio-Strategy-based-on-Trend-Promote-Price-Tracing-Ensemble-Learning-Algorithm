import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


from utils import tools
import seaborn as sns
from statsmodels.api import OLS
from scipy.optimize import curve_fit
from scipy.stats import chi2

class PickleMixin(object):

    def save(self, filename):
        """ Save object as a pickle """
        print("filename:",filename)
        # filename = "D:\PyCharm 2019.2.6\\awhole_project\other_algo\other_algo\\resultSave\sp500_PPT"
        file1 = open(filename, "wb")
        pickle.dump(self, file1, -1)
        file1.close()

    @classmethod
    def load(cls, filename):
        """ Load pickled object. """
        with open(filename, 'rb') as f:
            return pickle.load(f)


def least_squares_function(x, a, b, epsilon):
    return a + b * x + epsilon

class AlgoResult(PickleMixin):
    """ Results returned by algo's run method. The class containts useful
    metrics such as sharpe ratio, mean return, drawdowns, ... and also
    many visualizations.
    You can specify transactions by setting AlgoResult.fee. Fee is
    expressed in a percentages as a one-round fee.
    """

    def __init__(self, X, B):
        """
        :param X: Price relatives. ratio: all data divide the first day' data
        :param B: Weights.  # yjf. balances of all periods
        """
        # set initial values
        # 交易成本
        self._fee = 0.
        self._B = B
        self.rf_rate = 0.
        self._X = X

        # print("self.market_inf()初始化")
        # print(self.market_inf())
        # self.r_m = self.market_inf()

        # update logarithms, fees, etc.
        self._recalculate()

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, _X):
        self._X = _X
        self._recalculate()

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B
        self._recalculate()

    @property
    def fee(self):
        return self._fee

    @fee.setter
    def fee(self, value):
        """ Set transaction costs. Fees can be either float or Series
        of floats for individual assets with proper indices. """
        if isinstance(value, dict):
            value = pd.Series(value)
        if isinstance(value, pd.Series):
            missing = set(self.X.columns) - set(value.index)
            assert len(missing) == 0, 'Missing fees for {}'.format(missing)

        self._fee = value
        self._recalculate()

    def _recalculate(self):
        # calculate return for individual stocks
        # 重新计算CW
        # 计算个股收益
        # r为每一时期的股票的按投资比例投资的超额收益
        # self.asset_t为每一时期的每只股票的单位投资的收益
        # self.r每个时期的单位投资的收益
        # print("(self.X-1)[0:10]:",(self.X-1)[0:10])
        # print("B[0,10]:",self.B[0:10])
        r = (self.X - 1) * self.B
        # print("以下为单位投资的超额收益")
        # print(r[0:10])
        self.asset_r = r + 1
        # print("单位投资的收益")
        # print(self.asset_r[0:10])
        self.r = r.sum(axis=1) + 1
        self.r[0:4] = 1
        # print('self.r:',self.r)

        # stock went bankrupt
        # 股票破产
        self.r[self.r < 0] = 0.

        # add fees
        # 增加交易成本
        if not isinstance(self._fee, float) or self._fee != 0:
            fees = (self.B.shift(-1).mul(self.r, axis=0) - self.B * self.X).abs()
            fees.iloc[0] = self.B.ix[0]
            # fees.iloc[0] = self.B.iloc[0]  #dzw
            fees.iloc[-1] = 0.
            fees *= self._fee

            self.asset_r -= fees
            self.r -= fees.sum(axis=1)

        # 每个时期的单位投资的收益取对数
        self.r_log = np.log(self.r)


    @property
    def weights(self):
        return self.B

    @property
    def equity(self):
        return self.r.cumprod()


    @property
    def equity_decomposed(self):
        """ Return equity decomposed to individual assets. """
        return self.asset_r.cumprod()

    @property
    def asset_equity(self):
        return self.X.cumprod()

    @property
    def total_wealth(self):
        # print("self.asset_r.cumprod()")
        # print(self.r.cumprod())
        return self.r.prod()

    @property
    def MER(self):
        self.r_m = self.market_inf()
        # print("MER初始化后的")
        # print(self.r_m)
        r_algo = self.r - np.ones(len(self.r))
        r_market = self.r_m -np.ones(len(self.r))

        # return sum(self.r-self.r_m)/len(self.r)
        return sum(r_algo - r_market)/len(self.r)

    @property
    def sharpe_ratio(self):
        r_f = 0
        #   此处使用的期望为正态分布的期望
        r_algo = self.r - np.ones(len(self.r))
        # r_market = self.r_m - np.ones(len(self.r))
        print("means:", np.mean(r_algo))
        print("std:", np.std(r_algo))
        return ((np.mean(r_algo)-r_f) / np.std(r_algo))

    @property
    def information_ratio(self):
        return np.mean(self.r - self.r_m) / np.std(self.r - self.r_m)

    @property
    def profit_factor(self):
        x = self.r_log
        up = x[x > 0].sum()
        down = -x[x < 0].sum()
        return up / down if down != 0 else np.inf

    @property
    def sharpe(self):
        """ Compute annualized sharpe ratio from log returns. If data does
        not contain datetime index, assume daily frequency with 252 trading days a year.
        """
        return tools.sharpe(self.r_log, rf_rate=self.rf_rate, freq=self.freq())

    @property
    def p_value(self):
        p0 = [1, 1, 0]
        # 返回最优参数和协方差矩阵
        popt, pcov = curve_fit(least_squares_function, self.r_m, self.r, p0)
        print("popt:",popt)

        chi2_statistic = sum(
            (residual ** 2) / sigma_i ** 2 for residual, sigma_i in zip(self.r - least_squares_function(self.r_m, *popt), pcov[2]))
        p_value = 1 - chi2.cdf(chi2_statistic, 2)
        return p_value

    @property
    def information(self):
        """ Information ratio benchmarked against uniform CRP portfolio. """
        s = self.X.mean(axis=1)
        x = self.r_log - np.log(s)

        mu, sd = x.mean(), x.std()

        freq = self.freq()
        if sd > 1e-8:
            return mu / sd * np.sqrt(freq)
        elif mu > 1e-8:
            return np.inf * np.sign(mu)
        else:
            return 0.

    @property
    def ucrp_sharpe(self):
        # !!!要用哪个算法算夏普率
        from algos import CRP
        from algos import BAH
        result = BAH().run(self.X.cumprod())
        return result.sharpe
        pass

    @property
    def growth_rate(self):
        return self.r_log.mean() * self.freq()

    @property
    def volatility(self):
        return np.sqrt(self.freq()) * self.r_log.std()

    @property
    def annualized_return(self):
        return np.exp(self.r_log.mean() * self.freq()) - 1

    @property
    def annualized_volatility(self):
        return np.exp(self.r_log).std() * np.sqrt(self.freq())

    @property
    def drawdown_period(self):
        ''' Returns longest drawdown perid. Stagnation is a drawdown too. '''
        x = self.equity
        period = [0.] * len(x)
        peak = 0
        for i in range(len(x)):
            # new peak
            if x[i] > peak:
                peak = x[i]
                period[i] = 0
            else:
                period[i] = period[i-1] + 1
        return max(period) * 252. / self.freq()

    @property
    def max_drawdown(self):
        ''' Returns highest drawdown in percentage. '''
        x = self.equity
        return max(1. - x / x.cummax())

    @property
    def winning_pct(self):
        x = self.r_log
        win = (x > 0).sum()
        all_trades = (x != 0).sum()
        return float(win) / all_trades

    @property
    def turnover(self):
        return self.B.diff().abs().sum().sum()

    def freq(self, x=None):
        """ Number of data items per year. If data does not contain
        datetime index, assume daily frequency with 252 trading days a year."""
        x = x or self.r
        return tools.freq(x.index)

    def alpha_beta(self):
        rr = (self.X - 1).mean(1)

        m = OLS(self.r - 1, np.vstack([np.ones(len(self.r)), rr]).T)
        reg = m.fit()
        alpha, beta = reg.params.const * 252, reg.params.x1
        return alpha, beta

    def summary(self, name=None):
        alpha, beta = self.alpha_beta()

        return """Summary{}:
    Profit factor: {:.2f}
    Sharpe ratio: {:.2f}
    Information ratio(wrt UCRP): {: .2f}
    UCRP sharpe: {:.2f}
    Beta / Alpha: {:.2f} / {:.3%}
    Annualized return: {:.2%}
    Annualized volatility: {:.2%}
    Longest drawdown: {:.0f} days
    Max drawdown: {:.2%}
    Winning days: {:.1%}
    Turnover: {:.1f}
    Total wealth: {:.2%}
        """.format(
            '' if name is None else ' for ' + name,
            self.profit_factor,
            self.sharpe,
            self.information,
            self.ucrp_sharpe,
            beta,
            alpha,
            self.annualized_return,
            self.annualized_volatility,
            self.drawdown_period,
            self.max_drawdown,
            self.winning_pct,
            self.turnover,
            self.total_wealth
            )


    def market_inf(self):
        from algos import BAH
        algo_base = BAH()
        # print("SELF._X",self._X)
        output = algo_base.run(self._X)
        return output.r

    def search(self,name=None):
        alpha, beta = self.alpha_beta()
        # print("r:", self.r)
        return """Search{}:
            Total wealth: {:.2f}
            MER: {:.4f}
            P-value: {:.4%}
            Sharpe ratio: {:.4f}
            Annualized return: {:.2%}
            Max drawdown: {:.2%}
            Winning days: {:.1%}
            Calmar Ratio:{:.2%}
            Beta / Alpha: {:.2f} / {:.3%}
                """.format(
            '' if name is None else ' for ' + name,
            self.total_wealth,
            self.MER,
            self.p_value,
            self.sharpe_ratio,
            self.annualized_return,
            self.max_drawdown,
            self.winning_pct,
            self.annualized_return/self.max_drawdown,
            beta,
            alpha,
        )

    def getreturncw(self):
        return self.total_wealth
    def plot(self, weights=True, assets=True, portfolio_label='PORTFOLIO', show_only_important=True, **kwargs):
        """ Plot equity of all assets plus our strategy.
        :param weights: Plot weights as a subplot.
        :param assets: Plot asset prices.
        :return: List of axes.
        """
        res = ListResult([self], [portfolio_label])
        if not weights:
            print("not weights")
            ax1 = res.plot(assets=assets, **kwargs)
            return [ax1]
        else:
            print("weights")
            if show_only_important:
                ix = self.B.abs().sum().nlargest(n=20).index
                B = self.B.loc[:, ix].copy()
                assets = B.columns if assets else False
                B['_others'] = self.B.drop(ix, 1).sum(1)
            else:
                B = self.B.copy()

            plt.figure(1)

            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            res.plot(assets=assets, ax=ax1,**kwargs)
            # yhg. add, 2020.9.23
            # ax1.legend(B.columns, loc="upper left", prop={'size': 6})

            ax2 = plt.subplot2grid((3, 1), (2, 0), sharex=ax1)

            # plot weights as lines
            if B.drop(['CASH'], 1, errors='ignore').values.min() < -0.01:
                B.sort_index(axis=1).plot(ax=ax2, ylim=(min(0., B.values.min()), max(1., B.values.max())),
                                          legend=False, color=_colors(len(assets) + 1))
            else:
                # fix rounding errors near zero
                if B.values.min() < 0:
                    pB = B - B.values.min()
                else:
                    pB = B
                pB.sort_index(axis=1).plot(ax=ax2, ylim=(0., max(1., pB.sum(1).max())),
                                           legend=False, color=_colors(len(assets) + 1), kind='area', stacked=True)
            plt.ylabel('weights')
            return [ax1, ax2]

    def hedge(self, result=None):
        """ Hedge results with results of other strategy (subtract weights).
        :param result: Other result object. Default is UCRP.
        :return: New AlgoResult object.
        """
        if result is None:
            from algos import CRP
            result = CRP().run(self.X.cumprod())

        return AlgoResult(self.X, self.B - result.B)

    def plot_decomposition(self, **kwargs):
        """ Decompose equity into components of individual assets and plot
        them. Does not take fees into account. """
        ax = self.equity_decomposed.plot(**kwargs)
        return ax

    @property
    def importance(self):
        ws = self.weights.sum()
        return (ws / sum(ws)).order(ascending=False)

    def plot_total_weights(self):
        _, axes = plt.subplots(ncols=2)
        self.B.iloc[-1].sort_values(ascending=False).iloc[:15].plot(kind='bar', title='Latest weights', ax=axes[1])
        self.B.sum().sort_values(ascending=False).iloc[:15].plot(kind='bar', title='Total weights', ax=axes[0])


class ListResult(list, PickleMixin):
    """ List of AlgoResults. """

    def __init__(self, results=None, names=None):
        results = results if results is not None else []
        names = names if names is not None else []
        super(ListResult, self).__init__(results)
        self.names = names

    def append(self, result, name):
        super(ListResult, self).append(result)
        self.names.append(name)

    def to_dataframe(self):
        """ Calculate equities for all results and return one dataframe. """
        eq = {}
        for result, name in zip(self, self.names):
            eq[name] = result.equity
        return pd.DataFrame(eq)

    def save(self, filename, **kwargs):
        # do not save it with fees
        #self.fee = 0.
        #self.to_dataframe().to_pickle(*args, **kwargs)

        with open(filename, 'wb') as f:
            pickle.dump(self, f, -1)

    @classmethod
    def load(cls, filename):
        # df = pd.read_pickle(*args, **kwargs)
        # return cls([df[c] for c in df], df.columns)

        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def fee(self):
        return {name: result.fee for result, name in zip(self, self.names)}

    @fee.setter
    def fee(self, value):
        for result in self:
            result.fee = value

    def summary(self):
        return '\n'.join([result.summary(name) for result, name in zip(self, self.names)])

    def plot(self, ucrp=False, bah=False, assets=False, **kwargs):
        """ Plot strategy equity.
        :param ucrp: Add uniform CRP as a benchmark.
        :param bah: Add Buy-And-Hold portfolio as a benchmark.
        :param assets: Add asset prices.
        :param kwargs: Additional arguments for pd.DataFrame.plot
        """
        # NOTE: order of plotting is important because of coloring
        # plot portfolio
        print("bah:",bah)
        print("ucrp:",ucrp)
        d = self.to_dataframe()
        portfolio = d.copy()
        print("self.to_dataframe():\n",portfolio)
        ax = portfolio.plot(linewidth=1., legend=False, color='r', **kwargs)
        kwargs['ax'] = ax

        # print('cols: ', portfolio.columns)
        ax.set_ylabel('Total wealth')
        plt.show()

        # plot uniform constant rebalanced portfolio
        if ucrp:
            from algos.crp import CRP
            crp_algo = CRP().run(self[0].X.cumprod())
            crp_algo.fee = self[0].fee
            d['UCRP'] = crp_algo.equity
            d[['UCRP']].plot(**kwargs)

        # add bah
        if bah:
            from algos.bah import BAH
            bah_algo = BAH().run(self[0].X.cumprod())
            bah_algo.fee = self[0].fee
            d['BAH'] = bah_algo.equity
            d[['BAH']].plot(**kwargs)
        print(portfolio)
        plt.show()
        # add individual assets
        if isinstance(assets, bool):
            if assets:
                assets = self[0].asset_equity.columns
            else:
                assets = []

        if list(assets):
            self[0].asset_equity.sort_index(axis=1).plot(color=_colors(len(assets) + 1), **kwargs)

        # plot portfolio again to highlight it
        kwargs['color'] = 'blue'
        portfolio.plot(linewidth=3., **kwargs)

        return ax


def _colors(n):
    return sns.color_palette(n_colors=n)
