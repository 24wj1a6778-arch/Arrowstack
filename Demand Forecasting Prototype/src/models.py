"""Forecasting models (statsmodels / Prophet alternatives, written from scratch).

Why not statsmodels or Prophet?  The handbook allows documented alternatives.
These implementations are short, fully tested and need no extra installs, so
the project runs on any laptop.  statsmodels' ExponentialSmoothing is the drop-in
upgrade for HoltWinters (see README, "Next improvements").

Common interface
    model.fit(y, promo)                 y, promo: pd.Series with a daily DatetimeIndex
    model.predict(horizon, future_promo)  -> numpy array of non-negative forecasts

A model only ever receives data up to the forecast origin.  Planned promotions
for the horizon are passed separately (they are known in advance).

Models, from simplest to most complex
    naive_last          tomorrow = today
    moving_average_28   flat average of the last 28 days
    seasonal_naive      same weekday last week            <- the baseline to beat
    seasonal_mean       average of the same weekday over the last 4 weeks
    holt_winters        damped-trend exponential smoothing + weekly seasonality
    gbm_direct          gradient boosting on lag / calendar / promo features
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import config
from src.features import FEATURE_NAMES, direct_feature_matrix

M = config.SEASON


class BaseForecaster:
    name = "base"

    def fit(self, y: pd.Series, promo: pd.Series = None):
        raise NotImplementedError

    def predict(self, horizon: int, future_promo=None) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _clean(y):
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            raise ValueError("cannot fit on an empty history")
        return y


class NaiveLast(BaseForecaster):
    name = "naive_last"

    def fit(self, y, promo=None):
        self.last_ = float(self._clean(y)[-1])
        return self

    def predict(self, horizon, future_promo=None):
        return np.full(horizon, self.last_)


class MovingAverage(BaseForecaster):
    name = "moving_average_28"

    def __init__(self, window=28):
        self.window = window

    def fit(self, y, promo=None):
        self.mean_ = float(self._clean(y)[-self.window:].mean())
        return self

    def predict(self, horizon, future_promo=None):
        return np.full(horizon, self.mean_)


class SeasonalNaive(BaseForecaster):
    name = "seasonal_naive"

    def fit(self, y, promo=None):
        y = self._clean(y)
        if len(y) < M:
            raise ValueError(f"need at least {M} days of history")
        self.y_ = y
        return self

    def predict(self, horizon, future_promo=None):
        n = len(self.y_)
        return np.array([self.y_[n - M + (h - 1) % M] for h in range(1, horizon + 1)])


class SeasonalMean(BaseForecaster):
    name = "seasonal_mean"

    def __init__(self, weeks=4):
        self.weeks = weeks

    def fit(self, y, promo=None):
        y = self._clean(y)
        if len(y) < M:
            raise ValueError(f"need at least {M} days of history")
        self.y_ = y
        return self

    def predict(self, horizon, future_promo=None):
        n = len(self.y_)
        out = []
        for h in range(1, horizon + 1):
            base = n - M + (h - 1) % M
            vals = [self.y_[base - M * j] for j in range(self.weeks) if base - M * j >= 0]
            out.append(np.mean(vals))
        return np.array(out)


class HoltWinters(BaseForecaster):
    """Additive damped-trend Holt-Winters (ETS A,Ad,A) with weekly seasonality.

    Smoothing parameters are picked by a small grid search that minimises the
    one-step-ahead squared error on the last 52 weeks.  It uses no promotions
    or yearly seasonality - those limits show up in the error analysis.
    """
    name = "holt_winters"
    ALPHAS = (0.05, 0.1, 0.2, 0.3)
    BETAS = (0.0, 0.005, 0.02)
    GAMMAS = (0.0, 0.05, 0.15, 0.3)
    PHI = 0.95
    WINDOW = 52 * M

    @staticmethod
    def _run(y, alpha, beta, gamma, phi):
        level = y[:M].mean()
        trend = (y[M:2 * M].mean() - level) / M
        season = [y[i] - level for i in range(M)]
        sse = 0.0
        for t in range(M, len(y)):
            s = t % M
            pred = level + phi * trend + season[s]
            e = y[t] - pred
            sse += e * e
            level = level + phi * trend + alpha * e
            trend = phi * trend + beta * e
            season[s] = season[s] + gamma * e
        return sse, level, trend, season

    def fit(self, y, promo=None):
        y = self._clean(y)[-self.WINDOW:]
        if len(y) < 4 * M:
            raise ValueError(f"need at least {4 * M} days of history")
        best = None
        for a in self.ALPHAS:
            for b in self.BETAS:
                if b > a:
                    continue
                for g in self.GAMMAS:
                    res = self._run(y, a, b, g, self.PHI)
                    if best is None or res[0] < best[0]:
                        best = (*res, (a, b, g))
        _, self.level_, self.trend_, self.season_, self.params_ = best
        self.n_ = len(y)
        return self

    def predict(self, horizon, future_promo=None):
        out, damp = [], 0.0
        for h in range(1, horizon + 1):
            damp += self.PHI ** h
            out.append(self.level_ + damp * self.trend_ + self.season_[(self.n_ + h - 1) % M])
        return np.clip(np.array(out), 0, None)


class GBMDirect(BaseForecaster):
    """Direct multi-horizon gradient boosting.

    One model learns "sales h days ahead" from features known at the origin
    (recent averages, same-weekday history, last year's level) plus the target
    day's calendar and planned promotion.  Poisson loss suits count data and
    keeps predictions positive.
    """
    name = "gbm_direct"

    def __init__(self, seed=config.SEED, origin_step=config.GBM_ORIGIN_STEP):
        self.seed = seed
        self.origin_step = origin_step

    def fit(self, y, promo=None):
        yv = self._clean(y.values if hasattr(y, "values") else y)
        n = len(yv)
        if n < 150:
            raise ValueError("gbm_direct needs at least 150 days of history")
        dates = y.index
        pr = np.zeros(n) if promo is None else np.asarray(promo, dtype=float)
        last = n - 1
        origins, horizons = [], []
        for h in range(1, config.HORIZON + 1):
            o = np.arange(28, last - h + 1, self.origin_step)
            origins.append(o)
            horizons.append(np.full(len(o), h))
        origins, horizons = np.concatenate(origins), np.concatenate(horizons)
        X = direct_feature_matrix(yv, pr, dates, origins, horizons)
        target = yv[origins + horizons]
        self.model_ = HistGradientBoostingRegressor(
            loss="poisson", learning_rate=0.06, max_iter=150, max_leaf_nodes=12,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=False,
            random_state=self.seed)
        self.model_.fit(X, target)
        self.y_, self.promo_, self.dates_ = yv, pr, dates
        return self

    def predict(self, horizon, future_promo=None):
        if horizon > config.HORIZON:
            raise ValueError(f"gbm_direct was trained for horizons up to {config.HORIZON}")
        n = len(self.y_)
        fut = np.zeros(horizon) if future_promo is None else np.asarray(future_promo, dtype=float)
        dates = self.dates_.append(pd.date_range(self.dates_[-1] + pd.Timedelta(days=1), periods=horizon))
        promo = np.concatenate([self.promo_, fut])
        horizons = np.arange(1, horizon + 1)
        X = direct_feature_matrix(self.y_, promo, dates, np.full(horizon, n - 1), horizons)
        return np.clip(self.model_.predict(X), 0, None)


MODEL_FACTORIES = {
    "naive_last": NaiveLast,
    "moving_average_28": MovingAverage,
    "seasonal_naive": SeasonalNaive,
    "seasonal_mean": SeasonalMean,
    "holt_winters": HoltWinters,
    "gbm_direct": GBMDirect,
}
BASELINE = "seasonal_naive"
