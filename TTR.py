#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 16:33:30 2017
@author: https://github.com/Quantmatic/
"""
import math
import pandas as pd


def sma(series, per=30):
    ''' Simple Moving Average '''
    sma = pd.Series(series.rolling(per).mean(), name='SMA_'+str(per))
    return sma

def ema(series, per=30):
    ''' Exponential Moving Average '''
    ema = series.ewm(ignore_na=True, span=per, min_periods=per-1).mean()
    return pd.Series(ema, name='EMA_'+str(per))

def tema(series, per=30):
    ''' T3, Triple EMA '''
    tema = (3 * ema(series, per)) - (3 * ema(ema(series, per), per)) + \
           ema(ema(ema(series, per), per), per)
    return pd.Series(tema, name='T3_'+str(per))

def momentum(series, per=30):
    ''' Momentum '''
    mom = pd.Series(series.diff(per), name='MOM_'+str(per))
    return mom

def force(_df, per=14):
    ''' Force Index '''
    frc = pd.Series(_df.close.diff(per) * _df.volume.diff(per), name='Force_' + str(per))
    return frc

def stddev(series, per=14):
    ''' Standard Deviation '''
    stddev = series.rolling(window=per, center=False).std()
    return pd.Series(stddev, name='STD_' + str(per))

def rsi(series, per=30):
    ''' Relative Strength Index '''
    delta = series.diff()
    uuu = delta * 0
    ddd = uuu.copy()
    i_pos = delta > 0
    i_neg = delta < 0
    uuu[i_pos] = delta[i_pos]
    ddd[i_neg] = delta[i_neg]
    rsi = uuu.ewm(ignore_na=True, span=per, min_periods=0, adjust=True).std() / \
          ddd.ewm(ignore_na=True, span=per, min_periods=0, adjust=True).std()

    res = pd.Series(100 - 100 / (1 + rsi), name='RSI_'+str(per))
    return res

def kaufman(series, per=30):
    ''' Kaufman efficiency ratio '''
    direction = series.diff(per).abs()
    volatility = series.diff().abs().rolling(window=per, center=False).sum()
    kaufman = direction/volatility
    return pd.Series(kaufman, name='kaufman_'+str(per))

def garman_klass(_df, per=30):
    ''' Garman Klass volatility '''
    log_hl = (_df.high / _df.low).apply(math.log)
    log_co = (_df.close / _df.open).apply(math.log)
    rs_ = 0.5 * log_hl**2 - (2*math.log(2)-1) * log_co**2

    def _func(vvv):
        ''' math square root '''
        return math.sqrt(252 * vvv)

    res = rs_.rolling(per, center=False).mean().apply(func=_func)
    return pd.Series(res, name='vGK_'+str(per))

def rogers_satchell(_df, per=30):
    ''' Rogers Satchell Volatility '''
    log_ho = (_df.high / _df.open).apply(math.log)
    log_lo = (_df.low / _df.open).apply(math.log)
    log_co = (_df.close / _df.open).apply(math.log)

    rs_ = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def _func(vvv):
        ''' math square root '''
        return math.sqrt(252 * vvv)

    res = rs_.rolling(window=per, center=False).mean().apply(func=_func)
    return pd.Series(res, name='vRS_'+str(per))

def yang_zhang(_df, per=30):
    ''' Yang Zhang Volatility '''
    log_ho = (_df.high / _df.open).apply(math.log)
    log_lo = (_df.low / _df.open).apply(math.log)
    log_co = (_df.close / _df.open).apply(math.log)

    log_oc = (_df.open / _df.close.shift(1)).apply(math.log)
    log_oc_sq = log_oc**2

    log_cc = (_df.close / _df.close.shift(1)).apply(math.log)
    log_cc_sq = log_cc**2

    rs_ = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window=per, center=False).sum() * (1.0 / (per - 1.0))
    open_vol = log_oc_sq.rolling(window=per, center=False).sum() * (1.0 / (per - 1.0))
    window_rs = rs_.rolling(window=per, center=False).sum() * (1.0 / (per - 1.0))

    res = (open_vol + 0.164333 * close_vol + 0.835667 * window_rs).apply(math.sqrt) * math.sqrt(252)
    return pd.Series(res, name='vYZ_'+str(per))

def cmf(_df, per=30):
    ''' Chaikin Money Flow '''
    vol = _df.volume
    clv = ((_df.close - _df.low)-(_df.high - _df.close)) / (_df.high - _df.low) * vol
    ret = pd.Series(clv.rolling(per, center=False).sum() / \
                    vol.rolling(per, center=False).sum(), name='CMF_'+str(per))
    return ret

def cho(_df, per1, per2):
    ''' Chaikin Oscillator '''
    _ad = (2 * _df.close - _df.high - _df.low) / (_df.high - _df.low) * _df.volume
    ret = _ad.ewm(ignore_na=False, span=per1, min_periods=2, adjust=True).mean() - \
          _ad.ewm(ignore_na=False, span=per2, min_periods=9, adjust=True).mean()
    chaikin = pd.Series(ret, name='CHO_'+str(per1))
    return chaikin

def roc(series, per=10):
    ''' rate of change '''
    mmm = series.diff(per-1)
    nnn = series.shift(per-1)
    roc = pd.Series(mmm / nnn, name='ROC_'+str(per))
    return roc

def bollinger(series, per, dev):
    ''' Bollinger bands '''
    bbperiod = per
    bbndev = dev
    bbmid = series.rolling(window=bbperiod, center=False).mean()
    bbstd = series.rolling(window=bbperiod, center=False).std()
    bbupper = pd.Series(bbmid + bbndev * bbstd, name='bbupper')
    bblower = pd.Series(bbmid - bbndev * bbstd, name='bblower')
    return bbupper, bblower, pd.Series(bbmid, name='bbmid')

def donchian(_df, per=30):
    ''' Donchian Channel '''
    dcuarr = _df.high.rolling(window=per, center=False).max()
    dclarr = _df.low.rolling(window=per, center=False).min()

    d_upper = pd.Series(dcuarr, name='donch_upper')
    d_lower = pd.Series(dclarr, name='donch_lower')
    return d_upper, d_lower

def mfi(_df, per=30):
    ''' Money Flow Index and Ratio '''
    hlc = (_df.high + _df.low + _df.close)/3
    posmf = pd.Series(0.0, index=_df.index)
    negmf = pd.Series(0.0, index=_df.index)
    total_mf = hlc * _df.volume
    mask1 = (hlc > hlc.shift(1))
    mask2 = (hlc < hlc.shift(1))
    posmf[mask1] = total_mf[mask1]
    negmf[mask2] = total_mf[mask2]
    mfr = posmf / negmf

    mfi = 100 - (100 / (1 + mfr))
    mfi = pd.Series(mfi.rolling(window=per, center=False).mean(), name='MFI_' + str(per))
    return mfi

def dvo(_df, per=126, smooth=2):
    '''DVO,  David Varadi, CSS analitics'''
    ratio = _df.close /((_df.high + _df.low) / 2)
    avgratio = sma(ratio, smooth)
    _dvo = avgratio.rolling(per).mean().rank(ascending=False, pct=True)*100
    dvo = _dvo + 50 - sma(_dvo, per)
    return pd.Series(dvo, name='DVO_' + str(per))

def dvi(series, per=250):
    '''DVI,  David Varadi, CSS analitics'''
    _xf = pd.Series(-1, index=series.index)
    mask = (series > series.shift(1))
    _xf[mask] = 1

    _xg = _xf.rolling(10).sum() + _xf.rolling(100).sum()/10
    _xh = sma(_xg, 2)
    stretch = _xh.rolling(per).mean().rank(ascending=False, pct=True)*100
    _xc = (series/sma(series, 3))-1
    _xd = (sma(_xc, 5) + sma(_xc, 100)/10)/2
    _xe = sma(_xd, 5)
    magnitude = _xe.rolling(per).mean().rank(ascending=False, pct=True)*100
    xdvi = (0.8*magnitude) + (0.2*stretch)
    return pd.Series(xdvi, name='DVI_'+str(per))

def dvo3(_df, per=126, smooth=10):
    '''DVO,  David Varadi, CSS analitics'''
    ratio = _df.close /((_df.high + _df.low) / 2)
    avgratio = sma(ratio, 2)
    _dvo = avgratio.rank(ascending=False, pct=True).rolling(smooth).mean()*100
    dvo = _dvo + 50 - sma(_dvo, per)
    return pd.Series(dvo, name='DVO_' + str(per))

def dvi3(series, per=10):
    ''' DVO,  David Varadi, CSS analitics '''
    _xf = pd.Series(-1, index=series.index)
    mask = (series > series.shift(1))
    _xf[mask] = 1

    _xg = _xf.rolling(10).sum() + _xf.rolling(100).sum()/10
    _xh = sma(_xg, 2)
    stretch = _xh.rank(ascending=False, pct=True).rolling(per).mean()*100
    _xc = (series/sma(series, 3))-1
    _xd = (sma(_xc, 5) + sma(_xc, 100)/10)/2
    _xe = sma(_xd, 5)
    magnitude = _xe.rank(ascending=False, pct=True).rolling(per).mean()*100
    xdvi = (0.8*magnitude) + (0.2*stretch)
    return pd.Series(xdvi, name='DV3_'+str(per))

def cci(_df, per=30):
    ''' Commodity Channel Index '''
    ppa = (_df.high + _df.low + _df.close) / 3
    cci = pd.Series((ppa - ppa.rolling(window=per, center=False).mean()) / \
                    ppa.rolling(window=per, center=False).std(), name='CCI_' + str(per))

    return cci
