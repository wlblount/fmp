# fmp
functions for financial modelling prep

Function: fmp_13F
Docstring:
Inputs: cik# as a string
            leng as a string = number of symbols to return
        date as string yyyy-mm-dd is one of 4 quarted end dates:  3/31, 6/30, 9/30, or 12/31.  will ususlly not be
        availabe until AFTER 45 days from quarter end
Output: top 40 holdings df of date of report, symbol, position size in shares
        and dollars, % of position, and calculated price
                and sorted by position percentage


----------------------------------------
Function: fmp_13Fcik
Docstring: Input:  cik number as a string
Output: Entity name as a string


----------------------------------------
Function: fmp_13Fentity
Docstring:
Inputs: entity name as string

Output: dataframe of enbtity name matches and cik #


----------------------------------------
Function: fmp_2SymReg
Docstring: None
----------------------------------------
Function: fmp_balts
Docstring:     facs: LIST = ['date', 'symbol', 'reportedCurrency', 'fillingDate', 'acceptedDate',
   'period', 'cashAndCashEquivalents', 'shortTermInvestments',
   'cashAndShortTermInvestments', 'netReceivables', 'inventory',
   'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet',
   'goodwill', 'intangibleAssets', 'goodwillAndIntangibleAssets',
   'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets',
   'totalNonCurrentAssets', 'otherAssets', 'totalAssets',
   'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue',
   'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt',
   'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
   'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities',
   'otherLiabilities', 'totalLiabilities', 'commonStock',
   'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss',
   'othertotalStockholdersEquity', 'totalStockholdersEquity',
   'totalLiabilitiesAndStockholdersEquity', 'totalInvestments',
   'totalDebt', 'netDebt']
period = quarter, year
----------------------------------------
Function: fmp_cap
Docstring: Gross Cap is 4 quarters rolling sum of Revenue divided by current quarter Enterprise Value
params sym: str
returns: Series with date index and Gross Cap
----------------------------------------
Function: fmp_cashfts
Docstring: ######stmt = income-statement, balance-sheet-statement, cash-flow-statement, enterprise-values####
    facs = ['date', 'symbol', 'reportedCurrency', 'fillingDate', 'acceptedDate', 'period', 'netIncome',
'depreciationAndAmortization', 'deferredIncomeTax', 'stockBasedCompensation',
'changeInWorkingCapital', 'accountsReceivables', 'inventory', 'accountsPayables',
'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities',
'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments',
'salesMaturitiesOfInvestments', 'otherInvestingActivites', 'netCashUsedForInvestingActivites',
'debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 'dividendsPaid', 'otherFinancingActivites',
'netCashUsedProvidedByFinancingActivities', 'effectOfForexChangesOnCash', 'netChangeInCash',
'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod', 'operatingCashFlow', 'capitalExpenditure',
'freeCashFlow', 'link', 'finalLink']

period = quarter, year
----------------------------------------
Function: fmp_close
Docstring: returns a 2xn list of dicts {date: date,  close:close} with default length of 1000
leng = last n closing prices
----------------------------------------
Function: fmp_const
Docstring: Input:  sym:str --input an etf or mutual fund
tickersOnly: bool, returns a list of tickers.  default is False
Output: returns a df of the following columns:
        'name', 'pct', 'price', 'country', 'updated'
----------------------------------------
Function: fmp_cormatrix
Docstring: None
----------------------------------------
Function: fmp_corr
Docstring: input sy1, sy2 and lbk
outputs corr coefficient
----------------------------------------
Function: fmp_cumret
Docstring: None
----------------------------------------
Function: fmp_div
Docstring: Declaration Date: This is the date on which the company's board of directors announces
the upcoming dividend payment. It signifies the company's intention to pay a dividend.

Ex-Dividend Date: (Index)  The ex-dividend date is the first day following the declaration
date on which a stock trades without the dividend included. If you purchase the stock on
or after this date, you are not entitled to receive the dividend.

Record Date: The record date, also known as the ownership date, is the date on which the
company determines the shareholders who are eligible to receive the dividend. You must be
a registered shareholder on the record date to receive the dividend.

Payment Date: This is the date on which the dividend is actually paid to the eligible
shareholders. It is the day when the dividend amount is credited to the shareholders'
accounts or mailed out as physical checks.

trailYield:  rolling last 4 dividends
currYield:  last dividend x 4

----------------------------------------
Function: fmp_divHist
Docstring: input: symbol as string
returns:  DataFrame of dividend $/share with ex-date as the index
----------------------------------------
Function: fmp_earnCal
Docstring: input: start and end dates like 'XXXX-xx-xx'
returns: a dataframe of symbols with date, time and estimates where available
----------------------------------------
Function: fmp_earnDateNext
Docstring: input (str) sym
returns next earnings date as a string or 'NA' if there's a KeyError or no valid date
----------------------------------------
Function: fmp_earnEst
Docstring: input:  sym:  as string
        period: as string 'annual' or 'quarter'
returns: DatFrame wiyh columns  'RevLow','RevHigh','RevAvg','EbitdaLow','EbitdaHigh','EbitdaAvg','EbitLow', 'EbitHigh',
           'EbitAvg','NetIncLow','NetIncHigh','NetIncAvg','SgaExpLow','SgaExpHigh','SgaExpAvg',
        'EpsAvg','EpsHigh','EpsLow','numRev','numEps'
----------------------------------------
Function: fmp_earnSym
Docstring: input: symbol as string
       n as int.  the number of quarters to return
returns:  historical and future earnings dates, times and estimates
----------------------------------------
Function: fmp_efficiency
Docstring: Gross Cap is 4 quarters rolling sum of G&A Exp  divided by current quarter Revenue
params sym: str
returns: Series with date index and Gross Cap
----------------------------------------
Function: fmp_entts
Docstring:     facs = ['symbol', 'date', 'stockPrice', 'numberOfShares', 'marketCapitalization',
'minusCashAndCashEquivalents', 'addTotalDebt', 'enterpriseValue']
period = quarter, year
----------------------------------------
Function: fmp_filings
Docstring: None
----------------------------------------
Function: fmp_growth
Docstring: "revenueGrowth",
"grossProfitGrowth",
"ebitgrowth",
"operatingIncomeGrowth",
"netIncomeGrowth",
"epsgrowth",
"epsdilutedGrowth",
"weightedAverageSharesGrowth",
"weightedAverageSharesDilutedGrowth",
"dividendsperShareGrowth",
"operatingCashFlowGrowth",
"freeCashFlowGrowth",
"tenYRevenueGrowthPerShare",
"fiveYRevenueGrowthPerShare",
"threeYRevenueGrowthPerShare",
"tenYOperatingCFGrowthPerShare",
"fiveYOperatingCFGrowthPerShare",
"threeYOperatingCFGrowthPerShare",
"tenYNetIncomeGrowthPerShare",
"fiveYNetIncomeGrowthPerShare",
"threeYNetIncomeGrowthPerShare",
"tenYShareholdersEquityGrowthPerShare",
"fiveYShareholdersEquityGrowthPerShare",
"threeYShareholdersEquityGrowthPerShare",
"tenYDividendperShareGrowthPerShare",
"fiveYDividendperShareGrowthPerShare",
"threeYDividendperShareGrowthPerShare",
"receivablesGrowth",
"inventoryGrowth",
"assetGrowth",
"bookValueperShareGrowth",
"debtGrowth",
"rdexpenseGrowth",
"sgaexpensesGrowth"
----------------------------------------
Function: fmp_idx
Docstring:
       syms: list of symbols (no duplicates)
       weights: list of DOLLAR EXPOSURES per $1.00 of starting capital; one per
                symbol (len(weights) == len(syms)). If None, equal LONG
                weights (1/N each) are used.

                HOW TO READ A WEIGHT
                  sign      -> direction: positive = LONG, negative = SHORT
                  magnitude -> size: dollars of exposure per $1 of capital.
                               0.50 = half the capital, 1.00 = fully invested,
                               2.00 = 2x levered (i.e. borrowed) exposure.
                Weights do NOT have to sum to 1. Two numbers describe any book:
                  sum(|w|) = GROSS exposure = leverage
                  sum(w)   = NET exposure   = market direction
                (both are printed on every run)

                COOKBOOK
                  [0.5, 0.5]    long-only, fully invested   (gross 1.0x, net +1.0x)
                  [0.4, 0.4]    long-only, 20% in cash      (gross 0.8x, net +0.8x)
                  [-1.0]        short 1x                    (gross 1.0x, net -1.0x)
                  [1.0, -1.0]   market-neutral pair         (gross 2.0x, net  0.0x)
                  [1.3, -0.3]   130/30 book                 (gross 1.6x, net +1.0x)
                  [2.0]         2x levered long             (gross 2.0x, net +2.0x)
                  [1.5, -0.5]   levered long/short tilt     (gross 2.0x, net +1.0x)

                In the bt route, weights summing to <1 leave the remainder in
                cash; summing to >1 implies borrowing. NEITHER route models
                financing, borrow fees, or margin costs -- levered and short
                results are gross of those.
       rebal:  rebalance frequency: 'once', 'weekly', 'monthly', 'quarterly', or 'yearly'.
                'once' = buy-and-hold (constant SHARES; effective weights drift).
                *** IMPORTANT for SHORTS / LONG-SHORT books ***
                'once' holds constant shares, so a short leg's dollar exposure
                balloons as the underlier rises: leverage drifts and a net-short
                NAV can cross zero, which makes res.prices.pct_change() volatility
                MEANINGLESS. Therefore, when any weight < 0 and rebal=='once', this
                function logs a WARNING to stderr (NOT silenced by verbose=False or
                redirect_stdout) and AUTO-SWITCHES to rebal='quarterly'.
                Detect it programmatically via res.rebal == 'quarterly'.
                Long-only books are unaffected (drift is minor) and 'once' is fine.
                More frequent rebalancing converges to the constant-weight return.
       return_stream: bool, default False.
                False -> returns the bt backtest Result object (res); NAV = res.prices.
                True  -> BYPASSES bt entirely and returns a pandas DataFrame with
                         columns ['nav', 'ret'] (NAV FIRST):
                           nav: constant-weight, DAILY-rebalanced NAV, base 100.
                                Row 0 is the first price date at exactly 100.0, so
                                nav.iloc[-1] / nav.iloc[0] - 1 is the total return.
                           ret: the daily portfolio return (row 0 is 0.0).
                         This is the EXACT, robust route for SHORT and LONG/SHORT
                         volatility / Sharpe work: it never crosses zero and carries no
                         constant-share leverage drift.
                         Annualized vol = ret.iloc[1:].std() * sqrt(252).
       fac:   price field to use (default 'adjClose' = dividend/split adjusted).
       start: start date 'YYYY-MM-DD' (data begins at the latest common first
              date across syms if later than this -- see effective_start below).
       end:   end date 'YYYY-MM-DD'; None (default) = today. Set start AND end
              to study a historical window (e.g. 2022 only).
       name:  a label for the index, string.
       verbose: bool, default True. Print the run summary (first/last date,
              weights, gross/net, type, final row). Pass False when looping over
              many baskets. Errors and the short/'once' auto-switch warning are
              never silenced.

       Reconciliation (RSP +1 / SPY -1, 2023-01 -> 2026-06):
         rebal='once'      -> 16.5% ann vol  (WRONG: NAV halved by share drift)
         rebal='quarterly' ->  7.4% ann vol  (correct)
         return_stream=True->  7.2% ann vol  (correct, exact constant-weight)
       Short SPY [-1]: 'once' drove NAV negative (vol 125%, garbage);
         'quarterly' / return_stream give the correct ~15%.

       Examples:
           # long-only equal-weight index, buy & hold
           res = fmp_idx(['SPY', 'QQQ', 'IWM'])
           # short a single name (auto-switches to quarterly)
           res = fmp_idx(['SPY'], weights=[-1.0])
           # market-neutral pair -- exact vol via the return stream
           ls = fmp_idx(['RSP', 'SPY'], weights=[1.0, -1.0], return_stream=True)
           ann_vol = ls['ret'].iloc[1:].std() * (252 ** 0.5) * 100
           # quiet loop over many baskets
           for b in baskets:
               r = fmp_idx(b, start='2024-01-01', return_stream=True, verbose=False)

       RUN METADATA (both routes) -- assert on these instead of parsing stdout:
           effective_start: pd.Timestamp of the first REAL price bar used
                            (the latest common first date across syms).
           effective_end:   pd.Timestamp of the last price bar used.
           syms, weights:   the constituent list and the weights actually applied.
         bt route:          plain attributes -> res.effective_start, res.syms, ...
         return_stream:     DataFrame.attrs  -> df.attrs['effective_start'], ...
         NOTE (bt route): bt prepends a synthetic base-100 row dated one CALENDAR
         day before the first price bar, so res.prices.index[0] can be a
         weekend. res.effective_start == res.prices.index[1] is the first
         trading day.

       returns (return_stream=False): a bt.backtest.Result (bt 1.2.x). It is a
       dict subclass keyed by strategy name. The REAL surface is:
         Attributes
           res.prices            DataFrame, one column (name) = index NAV, base 100
           res.stats             DataFrame of performance stats (CAGR, vol, Sharpe,
                                 max drawdown, ...); res.stats[name] for the Series
           res.lookback_returns  DataFrame of trailing-window returns
           res.backtests         {name: bt.Backtest}; res.backtests[name].strategy
                                 exposes the underlying strategy object
           res.backtest_list     same, as a list
         Methods
           res.display()                  print the stats table
           res.display_lookback_returns()
           res.display_monthly_returns()
           res.plot()                     equity curve
           res.plot_weights()             weights over time
           res.plot_security_weights()
           res.plot_histograms()
           res.get_security_weights()     DataFrame of per-security weights by date
           res.get_weights()
           res.get_transactions()         DataFrame of every fill
           res.set_date_range(start, end)
           res.set_riskfree_rate(rf)
           res.to_csv(path)
       Daily portfolio returns: res.prices[name].pct_change()
       Constituents:            list(res.get_security_weights().columns) or res.syms
       There is NO res.perf, res.rets, res.assets, res.benchmark,
       res.security_weights, or res.strategy on this object.
----------------------------------------
Function: fmp_incts
Docstring: ######stmt = income-statement, balance-sheet-statement, cash-flow-statement, enterprise-values####
    facs = ['date', 'symbol', 'reportedCurrency', 'fillingDate', 'acceptedDate',
   'period', 'revenue', 'costOfRevenue', 'grossProfit', 'grossProfitRatio',
   'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
   'sellingAndMarketingExpenses',
   'sellingGeneralAndAdministrativeExpenses', 'otherExpenses',
   'operatingExpenses', 'costAndExpenses', 'interestExpense',
   'depreciationAndAmortization', 'ebitda', 'ebitdaratio',
   'operatingIncome', 'operatingIncomeRatio',
   'totalOtherIncomeExpensesNet', 'incomeBeforeTax',
   'incomeBeforeTaxRatio', 'incomeTaxExpense', 'netIncome',
   'netIncomeRatio', 'eps', 'epsdiluted', 'weightedAverageShsOut',
   'weightedAverageShsOutDil'
period = quarter, year
----------------------------------------
Function: fmp_intra
Docstring: sym = single symbol as a string (not case sensitive)
period = '1day' returns:  'adjClose', 'change', 'changeOverTime', 'changePercent', 'close','high', 'label', 'low', 'open', 'unadjustedVolume', 'volume', 'vwap'
period = '1min', '5min', '15min', '30min', '1hour' returns:  'close', 'high', 'low', 'open', 'volume'
----------------------------------------
Function: fmp_isActive
Docstring: input: a string or a list of strings / symbols
prints the symbol if not actively trading or not in the FMP system
----------------------------------------
Function: fmp_keyMetrics
Docstring: facs=['revenuePerShare','netIncomePerShare','operatingCashFlowPerShare',
      'freeCashFlowPerShare','cashPerShare','bookValuePerShare',
      'tangibleBookValuePerShare','shareholdersEquityPerShare',
      'interestDebtPerShare','marketCap','enterpriseValue','peRatio',
      'priceToSalesRatio' 'pocfratio','pfcfRatio','pbRatio','ptbRatio',
      'evToSales','enterpriseValueOverEBITDA','evToOperatingCashFlow',
      'evToFreeCashFlow','earningsYield','freeCashFlowYield',
      'debtToEquity','debtToAssets','netDebtToEBITDA','currentRatio',
      'interestCoverage','incomeQuality','dividendYield','payoutRatio',
      'salesGeneralAndAdministrativeToRevenue','researchAndDdevelopementToRevenue',
      'intangiblesToTotalAssets','capexToOperatingCashFlow','capexToRevenue',
      'capexToDepreciation','stockBasedCompensationToRevenue','grahamNumber',
      'roic','returnOnTangibleAssets','grahamNetNet','workingCapital',
      'tangibleAssetValue', 'netCurrentAssetValue','investedCapital',
      'averageReceivables','averagePayables','averageInventory',
      'daysSalesOutstanding','daysPayablesOutstanding','daysOfInventoryOnHand',
      'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover',
      'roe','capexPerShare']
----------------------------------------
Function: fmp_keyMetricsttm
Docstring: facs=[['revenuePerShareTTM','netIncomePerShareTTM','operatingCashFlowPerShareTTM',
       'freeCashFlowPerShareTTM','cashPerShareTTM','bookValuePerShareTTM',
       'tangibleBookValuePerShareTTM','shareholdersEquityPerShareTTM','interestDebtPerShareTTM',
       'marketCapTTM', 'enterpriseValueTTM','peRatioTTM','priceToSalesRatioTTM',
       'pocfratioTTM','pfcfRatioTTM','pbRatioTTM','ptbRatioTTM','evToSalesTTM',
       'enterpriseValueOverEBITDATTM', 'evToOperatingCashFlowTTM', 'evToFreeCashFlowTTM',
       'earningsYieldTTM', 'freeCashFlowYieldTTM', 'debtToEquityTTM','debtToAssetsTTM',
       'netDebtToEBITDATTM', 'currentRatioTTM', 'interestCoverageTTM','incomeQualityTTM',
       'dividendYieldTTM','dividendYieldPercentageTTM','payoutRatioTTM',
       'salesGeneralAndAdministrativeToRevenueTTM','researchAndDevelopementToRevenueTTM',
       'intangiblesToTotalAssetsTTM','capexToOperatingCashFlowTTM',
       'capexToRevenueTTM','capexToDepreciationTTM','stockBasedCompensationToRevenueTTM',
       'grahamNumberTTM','roicTTM','returnOnTangibleAssetsTTM','grahamNetNetTTM',
       'workingCapitalTTM','tangibleAssetValueTTM','netCurrentAssetValueTTM',
       'investedCapitalTTM','averageReceivablesTTM','averagePayablesTTM',
       'averageInventoryTTM','daysSalesOutstandingTTM','daysPayablesOutstandingTTM',
       'daysOfInventoryOnHandTTM','receivablesTurnoverTTM','payablesTurnoverTTM',
       'inventoryTurnoverTTM','roeTTM','capexPerShareTTM','dividendPerShareTTM',
       'debtToMarketCapTTM']
----------------------------------------
Function: fmp_mcap
Docstring: input: symbols as a string
outputs: dataframe of marketcap values with a datetime index

----------------------------------------
Function: fmp_mergarb
Docstring: syms:  list of 2 symbols as strings.  acquirer first, acquired second.
       ex: ['CNI', 'KSU']
shar_fact: float, number of shareds of acquirer that acquired is receiving
cash: amount of cash per share acquired is recieving
per: bool.  True returns spread as percentage of acquired,
            False returns spread as a float

returns a df of acquirer price, acquired price, and Arb calculation with
columns sym1, sym2, and Arb

----------------------------------------
Function: fmp_news
Docstring: syms= symbols in the form of: 'KSHB,GNLN,PBL,NBN,SKT'
convert list of strings by: ','.join(['KSHB', 'GNLN', 'PBL'])
----------------------------------------
Function: fmp_newsdict
Docstring: syms= symbols in the form of: 'KSHB,GNLN,PBL,NBN,SKT' .  Max 5 symbol limit by FMP
convert list of strings by: ','.join(['KSHB', 'GNLN', 'PBL'])
returns a dict
----------------------------------------
Function: fmp_peers
Docstring: Input: Symbol or list of symbols
Returns:  sym plus a list of peer symbols
----------------------------------------
Function: fmp_perfStats
Docstring: input: Series or Datframe of prices
outputs: ab object displaying performance stats
uses ffn package
----------------------------------------
Function: fmp_plotBarRetts
Docstring: None
----------------------------------------
Function: fmp_plotDualAxis
Docstring: Plots the given DataFrame using dual y-axes.

:param df: The DataFrame to plot.
:param left_cols: A column name or a list of column names as strings to plot on the left y-axis.
:param right_cols: A column or a list of column names ast strings to plot on the right y-axis.
:param left_label: The label for the left y-axis as a string.
:param right_label: The label for the right y-axis as a string.

----------------------------------------
Function: fmp_plotFin
Docstring: parameters are data and title
----------------------------------------
Function: fmp_plotFinMult
Docstring: parameters are data and title
----------------------------------------
Function: fmp_plotMult
Docstring: None
----------------------------------------
Function: fmp_plotPiv
Docstring: None
----------------------------------------
Function: fmp_plotStackedRet
Docstring: This function plots 3 stacked Return plots over a Line plot.

Inputs:
    retSyms is a list of exactly 3 symbols.  default is 'SPY', 'QQQ', 'IWM'
    lineSym is a string of a single symbol. defaul is '^TNX' (10Y Treas)
    start is a string 'YYYY-mm-dd'. default is 3 Months
Returns:
    4 vertically stacked plots with the first three being vertical bar plots of price returns
    and the last being a line plot of price




----------------------------------------
Function: fmp_plot_ts
Docstring: plot timeseries ignoring date gaps

Params
------
ts : pd.DataFrame or pd.Series
step : int, display interval for ticks
figsize : tuple, figure size
title: str
----------------------------------------
Function: fmp_plotyc
Docstring: None
----------------------------------------
Function: fmp_price
Docstring: Historical Price - Single Symbol - Multiple Facs
sym: string for single symbol as in 'SPY' with no []
start/end = string like 'YYYY-mm-dd'
facs= returns any of: 'open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume',
'change', 'changePercent', 'vwap', 'label', 'changeOverTime' with facs as column names
returns: DF with facs as the columns
----------------------------------------
Function: fmp_priceLbk
Docstring: inputs sym: single symbol as a string,
       date: 'YYYY-mm-dd' as a string
       facs: "date", "open","high", "low", "close", "adjClose",
       "volume", "unadjustedVolume", "change", "changePercent",
       "vwap", "label", "changeOverTime"

----------------------------------------
Function: fmp_priceLoop
Docstring: Multi Symbols no limit on history.
sym: string or a list of strings
start/end = string like 'YYYY-mm-dd'
facs= returns one of: 'open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume',
'change', 'changePercent', 'vwap', 'label', 'changeOverTime' with facs as column names
returns df: for single sym column names are facs, for mult sym column names are sym:facs
----------------------------------------
Function: fmp_priceMult
Docstring: Historical Price limited to 1 year. - Mutiple Symbols - Single Fac
***Max 5 symbols... returns the 1st 5 symbols without error***
syms: list of strings separated by commas ['SPY,'TLT']
start/end = string like 'YYYY-mm-dd'
facs= returns any of: 'open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume',
'change', 'changePercent', 'vwap', 'label', 'changeOverTime' with facs as column names
returns: Df with syms as columns
----------------------------------------
Function: fmp_prof
Docstring: returns dataframe for a symbol or list of symbols
facs = facs = 'symbol', 'price', 'beta', 'volAvg', 'mktCap', 'lastDiv', 'range',
   'changes', 'companyName', 'currency', 'cik', 'isin', 'cusip', 'exchange',
   'exchangeShortName', 'industry', 'website', 'description', 'ceo', 'sector',
   'country', 'fullTimeEmployees', 'phone', 'address', 'city', 'state', 'zip',
   'dcfDiff', 'dcf', 'image', 'ipoDate', 'defaultImage', 'isEtf',
   'isActivelyTrading'

----------------------------------------
Function: fmp_profF
Docstring: Returns the full profile of a single symbol
facs = 'symbol', 'price', 'beta', 'volAvg', 'mktCap', 'lastDiv', 'range',
'changes', 'companyName', 'currency', 'cik', 'isin', 'cusip', 'exchange',
'exchangeShortName', 'industry', 'website', 'description', 'ceo', 'sector',
'country', 'fullTimeEmployees', 'phone', 'address', 'city', 'state', 'zip',
'dcfDiff', 'dcf', 'image', 'ipoDate', 'defaultImage', 'isEtf',
'isActivelyTrading'
----------------------------------------
Function: fmp_ratios
Docstring: factors=['currentRatio', 'quickRatio', 'cashRatio',
'daysOfSalesOutstanding', 'daysOfInventoryOutstanding',
'operatingCycle', 'daysOfPayablesOutstanding', 'cashConversionCycle',
'grossProfitMargin', 'operatingProfitMargin', 'pretaxProfitMargin',
'netProfitMargin', 'effectiveTaxRate', 'returnOnAssets',
'returnOnEquity', 'returnOnCapitalEmployed', 'netIncomePerEBT',
'ebtPerEbit', 'ebitPerRevenue', 'debtRatio', 'debtEquityRatio',
'longTermDebtToCapitalization', 'totalDebtToCapitalization',
'interestCoverage', 'cashFlowToDebtRatio', 'companyEquityMultiplier',
'receivablesTurnover', 'payablesTurnover', 'inventoryTurnover',
'fixedAssetTurnover', 'assetTurnover', 'operatingCashFlowPerShare',
'freeCashFlowPerShare', 'cashPerShare', 'payoutRatio',
'operatingCashFlowSalesRatio', 'freeCashFlowOperatingCashFlowRatio',
'cashFlowCoverageRatios', 'shortTermCoverageRatios',
'capitalExpenditureCoverageRatio', 'dividendPaidAndCapexCoverageRatio',
'dividendPayoutRatio', 'priceBookValueRatio', 'priceToBookRatio',
'priceToSalesRatio', 'priceEarningsRatio', 'priceToFreeCashFlowsRatio',
'priceToOperatingCashFlowsRatio', 'priceCashFlowRatio',
'priceEarningsToGrowthRatio', 'priceSalesRatio', 'dividendYield',
'enterpriseValueMultiple', 'priceFairValue']
----------------------------------------
Function: fmp_ratiosttm
Docstring: facs=['dividendYielTTM','dividendYielPercentageTTM','peRatioTTM',
'pegRatioTTM','payoutRatioTTM','currentRatioTTM','quickRatioTTM',
'cashRatioTTM','daysOfSalesOutstandingTTM','daysOfInventoryOutstandingTTM',
'operatingCycleTTM','daysOfPayablesOutstandingTTM','cashConversionCycleTTM',
'grossProfitMarginTTM','operatingProfitMarginTTM','pretaxProfitMarginTTM',
'netProfitMarginTTM','effectiveTaxRateTTM','returnOnAssetsTTM',
'returnOnEquityTTM','returnOnCapitalEmployedTTM','netIncomePerEBTTTM',
'ebtPerEbitTTM','ebitPerRevenueTTM','debtRatioTTM','debtEquityRatioTTM',
'longTermDebtToCapitalizationTTM','totalDebtToCapitalizationTTM','interestCoverageTTM',
'cashFlowToDebtRatioTTM','companyEquityMultiplierTTM','receivablesTurnoverTTM',
'payablesTurnoverTTM','inventoryTurnoverTTM','fixedAssetTurnoverTTM',
'assetTurnoverTTM','operatingCashFlowPerShareTTM','freeCashFlowPerShareTTM',
'cashPerShareTTM','operatingCashFlowSalesRatioTTM','freeCashFlowOperatingCashFlowRatioTTM',
'cashFlowCoverageRatiosTTM','shortTermCoverageRatiosTTM','capitalExpenditureCoverageRatioTTM',
'dividendPaidAndCapexCoverageRatioTTM','priceBookValueRatioTTM','priceToBookRatioTTM',
'priceToSalesRatioTTM','priceEarningsRatioTTM','priceToFreeCashFlowsRatioTTM',
'priceToOperatingCashFlowsRatioTTM','priceCashFlowRatioTTM','priceEarningsToGrowthRatioTTM',
'priceSalesRatioTTM','enterpriseValueMultipleTTM','priceFairValueTTM','dividendPerShareTTM']
----------------------------------------
Function: fmp_rsi
Docstring: Returns a pd.Series with the relative strength index.
----------------------------------------
Function: fmp_scaler
Docstring: None
----------------------------------------
Function: fmp_screen
Docstring: ex:  fmp_screen(county='US', marketCapMoreThan='1000000000 ', industry='Insurance—Life')

Parameters:

MoreThan, LowerThan

marketCap, price, beta, volume, dividend, isEtf, isActivelyTrading

Sectors and Industries

Basic Materials:


Other Industrial Metals & Mining
Specialty Chemicals
Paper & Paper Products
Agricultural Inputs
Copper
Gold
Steel
Chemicals
Building Materials

Communication Services:


Internet Content & Information
Telecom Services
Entertainment
Electronic Gaming & Multimedia
Publishing
Advertising Agencies

Consumer Cyclical:


Internet Retail
Auto Manufacturers
Luxury Goods
Home Improvement Retail
Restaurants
Footwear & Accessories
Travel Services
Apparel Retail
Specialty Retail
Entertainment
Lodging
Resorts & Casinos
Auto & Truck Dealerships
Auto Parts
Residential Construction
Packaging & Containers
Personal Services

Consumer Defensive:


Discount Stores
Household & Personal Products
Beverages—Non-Alcoholic
Beverages—Brewers
Tobacco
Beverages—Wineries & Distilleries
Confectioners
Packaged Foods
Farm Products
Grocery Stores
Food Distribution

Energy:


Oil & Gas Integrated
Oil & Gas E&P
Oil & Gas Midstream
Oil & Gas Equipment & Services
Oil & Gas Refining & Marketing

Financial Services:


Banks—Diversified
Insurance—Diversified
Credit Services
Banks—Regional
Capital Markets
Financial Data & Stock Exchanges
Asset Management
Insurance—Property & Casualty
Insurance Brokers
Insurance
Insurance—Life
Insurance—Reinsurance
Shell Companies
Mortgage Finance

Healthcare:


Diagnostics & Research
Drug Manufacturers—General
Healthcare Plans
Biotechnology
Medical Devices
Medical Instruments & Supplies
Drug Manufacturers—Specialty & Generic
Medical Care Facilities
Medical Distribution
Pharmaceutical Retailers
Health Information Services

Industrials:


Integrated Freight & Logistics
Specialty Industrial Machinery
Aerospace & Defense
Conglomerates
Railroads
Farm & Heavy Construction Machinery
Staffing & Employment Services
Specialty Business Services
Waste Management
Electrical Equipment & Parts
Engineering & Construction
Rental & Leasing Services
None
Building Products & Equipment
Trucking
Industrial Distribution
Consulting Services
Infrastructure Operations
Airports & Air Services
Airlines

Real Estate:


REIT—Industrial
REIT—Specialty
REIT—Retail
REIT—Healthcare Facilities
REIT—Diversified
Real Estate Services
REIT—Office
REIT—Residential

Technology:


Consumer Electronics
Software—Infrastructure
Semiconductors
Semiconductor Equipment & Materials
Communication Equipment
Software—Application
Information Technology Services
Computer Hardware
Electronic Components
Scientific & Technical Instruments
Solar
Telecom Services - Foreign

Utilities:


Utilities—Regulated Electric
Utilities—Diversified
Utilities—Regulated Gas
Utilities—Regulated Water
Utilities—Renewable
Utilities Regulated

        Country:  'US', 'CN', 'TW', 'FR', 'CH', 'NL', 'CA', 'JP', 'DK', 'IE', 'AU',
       'GB', 'DE', 'SG', 'BE', 'IN', 'BR', 'ZA', 'AR', 'ES', 'NO', 'HK',
       'IT', 'MX', 'BM', 'LU', 'SE', 'FI', 'CO', 'KR', 'ID', 'JE', 'IL',
       'PT', 'UY', 'CL', 'MC', 'CY', 'MA', 'KY', 'RU', 'PR', 'PH', 'IS',
       'TR', 'IM', 'TH', 'PA', 'PE', 'GG', 'Peru', 'AE', 'NZ', 'GR', 'CR',
       'MY', 'BB', 'BS', 'GA', 'JO', 'VG', 'DO', 'ZM', 'MT', 'CK', 'MN',
       'LT', 'MO', 'AI'

        exchange:  'MCE', 'Johannesburg', 'HKSE', 'NYSEArca', 'BATS Exchange', 'Shanghai', 'Mexico',
           'Milan', 'FGI', 'ASE', 'Shenzhen', 'AMEX', 'Athens', 'TSXV', 'New York Stock Exchange', 'Paris',
           'Lisbon', 'Toronto', 'New York Stock Exchange Arca', 'Oslo', 'NZSE', 'OSL', 'Sao Paolo',
           'EURONEXT', 'MCX', 'Vienna', 'Nasdaq', 'NSE', 'SIX', 'OTC', 'Amsterdam', 'NASDAQ Global Market',
           'Taiwan', 'OSE', 'Nasdaq Capital Market', 'Helsinki', 'Other OTC', 'KSE', 'Santiago', 'Irish',
           'Canadian Sec', 'BATS', 'HKG', 'XETRA', 'NYSE American', 'Brussels', 'LSE', 'ASX', 'NASDAQ', 'YHD',
           'Frankfurt', 'NCM', 'Stockholm', 'Istanbul', 'Copenhagen', 'Nasdaq Global Market', 'Tokyo',
           'Nasdaq Global Select', 'Jakarta', 'KOSDAQ', 'SAT', 'Swiss', 'Hamburg', 'NMS'

----------------------------------------
Function: fmp_search
Docstring: for exchange searches:  ETF | MUTUAL_FUND | COMMODITY | INDEX | CRYPTO | FOREX | TSX | AMEX | NASDAQ | NYSE | EURONEXT
----------------------------------------
Function: fmp_sectInd
Docstring: Prints a Sector - Industry listing to the screen

----------------------------------------
Function: fmp_shares
Docstring: input:symbol
output: list : float, oustanding, percent float
----------------------------------------
Function: fmp_spread
Docstring: long: string long symbol
short: string short symbol
start/end = string like 'YYYY-mm-dd'
ratio: bool, long / short (default is long - short)
----------------------------------------
Function: fmp_stoch
Docstring: None
----------------------------------------
Function: fmp_ticker
Docstring: None
----------------------------------------
