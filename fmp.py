#version 1.0.5  updated 2/17/25 added acceptance date as a fmp_incts factor

    ##TO DO - fmp_screen():  explore if all financial factors work (bal sheet items, income, items, metrics???
    ##      - explore 'requote_uri' in all urls.
    ##      - look for unneeded imports

#     https://financialmodelingprep.com/developer/docs

from IPython.display import display, Markdown


import time
import certifi
import ssl
ssl_context = ssl.create_default_context(cafile=certifi.where())
import ffn
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime as dt
import json
import re
import logging
logging.captureWarnings(True)
from collections import defaultdict
from urllib.request import urlopen
from urllib.parse import urlencode
import requests
from datetime import datetime
from tqdm import notebook, tqdm
from requests.utils import requote_uri
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from IPython.display import display, HTML
import os
import bt
import webbrowser

from tvDatafeed import TvDatafeed, Interval
# Delay slow imports
def load_utils():
    import utils
    return utils


# Get the API key from the environment variable
apikey = os.getenv('FMP_API_KEY')

# Check if the API key is set
if not apikey:
    raise ValueError("API key not found. Please set the environment variable 'FMP_API_KEY'.")

pd.options.display.float_format = '{:,.2f}'.format

#-----------------------------------------------------  

def fmp_price(syms, start='1960-01-01', end=str(dt.datetime.now().date()), facs=['close']):
    '''
    Historical Price - Single Symbol - Multiple Facs
    sym: string for single symbol as in 'SPY' with no []
    start/end = string like 'YYYY-mm-dd'
    facs= returns any of: 'open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume', 
    'change', 'changePercent', 'vwap', 'label', 'changeOverTime' with facs as column names 
    returns: DF with facs as the columns
    '''
    
    
   
    url=requote_uri(f'https://financialmodelingprep.com/api/v3/historical-price-full/{syms}?from=     {start}&to={end}&apikey={apikey}')
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    # Updated fmp_price logic in fmp.py
    stuff = json.loads(data)

    # Safety Check: Does 'historical' exist?
    if 'historical' not in stuff:
    # Optional: Print a warning so you know which symbol failed
        print(f"Warning: No historical data found for symbol in response.") 
        return pd.DataFrame() # Return empty DF so the loop can continue

    l = stuff['historical']
    idx = [sub['date'] for sub in l]
    idx=pd.to_datetime(idx)
    df = pd.DataFrame([[sub[k] for k in facs] for sub in l], columns=facs, index=idx)
    return df.iloc[::-1]


#-----------------------------------------------------
def fmp_priceMult(syms, start='1960-01-01', end=str(dt.datetime.now().date()), fac='adjClose'):
    
    '''
    Historical Price limited to 1 year. - Mutiple Symbols - Single Fac
    ***Max 5 symbols... returns the 1st 5 symbols without error***
    syms: list of strings separated by commas ['SPY,'TLT']
    start/end = string like 'YYYY-mm-dd'
    facs= returns any of: 'open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume', 
    'change', 'changePercent', 'vwap', 'label', 'changeOverTime' with facs as column names 
    returns: Df with syms as columns
    '''
    syms=tuple(syms)
    syms=','.join(syms)

    url=requote_uri('https://financialmodelingprep.com/api/v3/historical-price-full/'+syms+'?from='+start+'&to='+end+'&apikey='+apikey)
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    data=stuff['historicalStockList'] 
    idx=[x['date'] for x in data[0]['historical']]
    idx=pd.to_datetime(idx)
    df = pd.DataFrame({d['symbol']: [x[fac] for x in d['historical']] for d in data}, 
                  index=idx)

    print ('fac= ', fac)
    return df.iloc[::-1]



#----------------------------------------------------------------------------


def fmp_priceLoop(syms, start='1960-01-01', end=str(dt.datetime.now().date()), fac='close', supress=True):
    '''
    Multi Symbols no limit on history.
    sym: string or a list of strings
    start/end = string like 'YYYY-mm-dd'
    facs= returns one of: 'open', 'high', 'low', 'close', 'adjClose', 'volume', 'unadjustedVolume', 
    'change', 'changePercent', 'vwap', 'label', 'changeOverTime' with facs as column names
    returns df: for single sym column names are facs, for mult sym column names are sym:facs
    '''
    df=pd.DataFrame()
    # In fmp.py inside fmp_priceLoop function

    if supress:
        for i in syms:
            dff = fmp_price(i, start=start, end=end, facs=[fac])
        
            # NEW: Check if dff is empty before assigning
            if not dff.empty:
                df[i] = dff
            else:
                print(f"Skipping {i}: No data returned.")

    else:
        # Do the same check for the non-supressed loop if you use it
        for i in notebook.tqdm(syms, disable=False):
            dff = fmp_price(i, start=start, end=end, facs=[fac])
            
            if not dff.empty:
                df[i] = dff
   
    return df

#-----------------------------------------------------

def fmp_priceLbk(sym, date,facs=['close']):  
    '''
    inputs sym: single symbol as a string,
           date: 'YYYY-mm-dd' as a string
           facs: "date", "open","high", "low", "close", "adjClose",
           "volume", "unadjustedVolume", "change", "changePercent",
           "vwap", "label", "changeOverTime"
           
    '''
    url= f"https://financialmodelingprep.com/api/v3/historical-price-full/{sym}?from={date}&to={date}&apikey=deb84eb89cd5f862f8f3216ea4d44719"
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    l=stuff['historical'] [0]
    return [l[key] for key in facs]
#--------------------------------------------------------    
def fmp_spread(long, short, start='1960-01-01', end=str(dt.datetime.now().date()), ratio=False):
    '''
    long: string long symbol
    short: string short symbol
    start/end = string like 'YYYY-mm-dd'
    ratio: bool, long / short (default is long - short)
    '''
    syms = [long, short]
    df=pd.DataFrame()
    
    if ratio:
        for i in syms:
            dff=fmp_price(i, start=start, end=end).pct_change().cumsum()
            df=pd.concat([df,dff], axis=1)
            df=df.dropna()
        df['spread'] = df.iloc[:,[0]]/df.iloc[:,[1]]
    else:
        for i in syms:
            dff=fmp_price(i, start=start, end=end).pct_change().cumsum()
            df=pd.concat([df,dff], axis=1)
            df=df.dropna()
        df['spread'] = df.iloc[:,[0]]-df.iloc[:,[1]]
    return df.spread




#-----------------------------------------------------

def fmp_mcap(sym):
    '''
input: symbols as a string
outputs: dataframe of marketcap values with a datetime index
    '''
    df=fmp_entts(sym, facs=['numberOfShares'])
    dfp=fmp_price(sym)
    dff = pd.concat([df, dfp], axis=1)
    dff.ffill(inplace=True)
    dff.dropna(inplace=True)
    dff['mktCap']=(dff.numberOfShares*dff.close) 
    return dff.mktCap

#-----------------------------------------------------
# 2026-02-08 08:20am

def fmp_balts(sym, facs=None, period='quarter', limit=8, save_md=None):
    '''
    Retrieves standardized Balance Sheet data from the FMP v3 endpoint.
    Displays FMP reported totals and validates against calculated component sums.
    
    Parameters:
    -----------
    sym : str
        The stock ticker symbol (e.g., 'AAPL').
    
    period : str, default='quarter'
        The reporting period to retrieve:
        - 'quarter' : Quarterly financial statements (default)
        - 'annual'  : Annual financial statements (FY only)
    
    limit : int, default=400
        Maximum number of periods to retrieve from FMP.
    
    facs : list, optional
        List of specific line items to return. If None, returns the full standard set.
        
        Mandatory Columns (Always Returned):
        ['reportedCurrency', 'calendarYear', 'period']
        
        AVAILABLE TAGS:
        Current Assets:
        ['cashAndCashEquivalents', 'shortTermInvestments', 'netReceivables', 
         'inventory', 'otherCurrentAssets']
        
        Non-Current Assets:
        ['propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets', 
         'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets']
        
        Current Liabilities:
        ['accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue', 
         'otherCurrentLiabilities']
        
        Non-Current Liabilities:
        ['longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
         'otherNonCurrentLiabilities', 'capitalLeaseObligations']
        
        Stockholders Equity:
        ['preferredStock', 'commonStock', 'retainedEarnings', 
         'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity',
         'minorityInterest']
        
        FMP Reported Totals (from 10-Q/10-K):
        ['totalCurrentAssets', 'totalNonCurrentAssets', 'totalAssets',
         'totalCurrentLiabilities', 'totalNonCurrentLiabilities', 'totalLiabilities',
         'totalStockholdersEquity', 'totalLiabilitiesAndStockholdersEquity']
        
        Validation Deltas (Reported - Calculated from Components):
        ['totalCurrentAssets_delta_rpt_vs_calc', 'totalNonCurrentAssets_delta_rpt_vs_calc',
         'totalAssets_delta_rpt_vs_calc', 'totalCurrentLiabilities_delta_rpt_vs_calc',
         'totalNonCurrentLiabilities_delta_rpt_vs_calc', 'totalLiabilities_delta_rpt_vs_calc',
         'totalStockholdersEquity_delta_rpt_vs_calc', 'totalLiabilitiesAndStockholdersEquity_delta_rpt_vs_calc',
         'accounting_equation_delta']
        
        Data Quality Flags:
        ['duplicate_period'] - True if multiple filings exist for same calendarYear/period
        
        Metadata:
        ['fillingDate']
    
    save_md : str, optional
        Filepath to save DataFrame with metadata as markdown file.
        If provided, saves output with company metadata header.
        Example: save_md='apple_balance_sheet.md'
    
    Returns:
    --------
    pd.DataFrame
        Balance sheet data with date index and symbol stored in df.attrs['symbol']
        Company metadata stored in df.attrs['metadata'] containing:
        ['companyName', 'cik', 'isin', 'cusip', 'currency', 'exchangeShortName',
         'industry', 'sector', 'country']
    
    LLM Context for Validation Columns:
    ------------------------------------
    This dataset includes validation columns (suffix: _delta_rpt_vs_calc) that verify 
    balance sheet internal consistency.
    
    Formula: Δ = FMP_reported_value - calculated_from_components
    
    Validation Columns & Their Formulas:
    - totalCurrentAssets_delta_rpt_vs_calc = totalCurrentAssets(rpt) - 
        (cashAndCashEquivalents + shortTermInvestments + netReceivables + inventory + otherCurrentAssets)
    - totalNonCurrentAssets_delta_rpt_vs_calc = totalNonCurrentAssets(rpt) - 
        (propertyPlantEquipmentNet + goodwill + intangibleAssets + longTermInvestments + taxAssets + otherNonCurrentAssets)
    - totalAssets_delta_rpt_vs_calc = totalAssets(rpt) - (totalCurrentAssets + totalNonCurrentAssets)
    - totalCurrentLiabilities_delta_rpt_vs_calc = totalCurrentLiabilities(rpt) - 
        (accountPayables + shortTermDebt + taxPayables + deferredRevenue + otherCurrentLiabilities)
    - totalNonCurrentLiabilities_delta_rpt_vs_calc = totalNonCurrentLiabilities(rpt) - 
        (longTermDebt + deferredRevenueNonCurrent + deferredTaxLiabilitiesNonCurrent + otherNonCurrentLiabilities + capitalLeaseObligations)
    - totalLiabilities_delta_rpt_vs_calc = totalLiabilities(rpt) - (totalCurrentLiabilities + totalNonCurrentLiabilities)
    - totalStockholdersEquity_delta_rpt_vs_calc = totalStockholdersEquity(rpt) - 
        (preferredStock + commonStock + retainedEarnings + accumulatedOtherComprehensiveIncomeLoss + othertotalStockholdersEquity + minorityInterest)
    - totalLiabilitiesAndStockholdersEquity_delta_rpt_vs_calc = totalLiabilitiesAndStockholdersEquity(rpt) - 
        (totalLiabilities + totalStockholdersEquity)
    - accounting_equation_delta = totalAssets - totalLiabilitiesAndStockholdersEquity
        (Fundamental accounting equation check: Assets = Liabilities + Equity)
    
    Interpreting Validation Deltas:
    Δ = 0        → Components reconcile perfectly with reported total ✓
    Δ > 0        → Reported total exceeds component sum
                   CAUSE: Component breakdown not disclosed in filing
                   EXAMPLE: Company reports "Other Current Assets" as single line without detail
    Δ < 0        → Component sum exceeds reported total
                   CAUSE: Potential data quality issue or parsing error 🚩
    Small Δ      → Rounding differences (typically <0.1% of total)
    
    Special Note on accounting_equation_delta:
    This should ALWAYS equal zero in valid financial statements.
    Non-zero values indicate serious data quality issues requiring manual review.
    
    Common Patterns:
    1. totalCurrentAssets_delta_rpt_vs_calc > 0 with otherCurrentAssets = 0
       → Company doesn't break out all current asset components (normal)
    
    2. totalStockholdersEquity_delta_rpt_vs_calc > 0 with many equity components = 0
       → Company uses simplified equity structure (common for many firms)
    
    3. accounting_equation_delta ≠ 0
       → Flag for immediate manual review 🚩
    
    When asked to verify data quality:
    1. Check accounting_equation_delta first (must be 0)
    2. For other deltas, check if delta = 0 (perfect reconciliation)
    3. If delta > 0, check if components exist (look for zeros)
    4. If components are zero, delta indicates missing breakdown (expected)
    5. If components exist but don't sum, flag as data quality issue
    6. Calculate delta as % of reported value to assess materiality
    
    Data Quality Flag - duplicate_period:
    - False: Single filing for this period (normal)
    - True: Multiple filings exist for same calendarYear/period combination
           CAUSES: Amended filings, restatements, or FMP data quality issues
           ACTION: Review fillingDate to identify most recent filing, or 
                   manually verify which filing to use
    
    Examples:
    ---------
    >>> # Get last 8 quarters
    >>> df = fmp_balts('AAPL', period='quarter', limit=8)
    
    >>> # Get last 5 years annual
    >>> df = fmp_balts('AAPL', period='annual', limit=5)
    
    >>> # Get specific fields only
    >>> df = fmp_balts('AAPL', facs=['totalAssets', 'totalLiabilities', 'totalStockholdersEquity'])
    
    >>> # Check for duplicate filings
    >>> df = fmp_balts('AZTA', period='quarter', limit=8)
    >>> if df['duplicate_period'].any():
    >>>     print("Warning: Multiple filings detected")
    >>>     print(df[df['duplicate_period']][['calendarYear', 'period', 'fillingDate']])
    
    >>> # Verify accounting equation
    >>> df = fmp_balts('AAPL')
    >>> if df['accounting_equation_delta'].abs().max() > 0:
    >>>     print("WARNING: Accounting equation doesn't balance!")
    
    >>> # Access company metadata
    >>> df = fmp_balts('AZTA')
    >>> print(df.attrs['metadata'])
    
    >>> # Save with metadata to markdown
    >>> df = fmp_balts('AZTA', save_md='azenta_balance_sheet.md')
    '''
    
    sym = sym.upper().strip()
    
    # Fetch company profile metadata
    try:
        profile = fmp_profF(sym)
        metadata = {
            'companyName': profile.get('companyName'),
            'cik': profile.get('cik'),
            'isin': profile.get('isin'),
            'cusip': profile.get('cusip'),
            'currency': profile.get('currency'),
            'exchangeShortName': profile.get('exchangeShortName'),
            'industry': profile.get('industry'),
            'sector': profile.get('sector'),
            'country': profile.get('country')
        }
    except Exception as e:
        print(f"Warning: Could not fetch metadata for {sym}: {e}")
        metadata = {}
    
    # URL construction
    url = f'https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}?period={period}&limit={limit}&apikey={apikey}'
    url = requote_uri(url)

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))
        if not stuff or not isinstance(stuff, list):
            return pd.DataFrame()

        df = pd.DataFrame(stuff)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = 'date'
        
        df = df.sort_index()
        
        # Drop columns that are 100% NaN
        df = df.dropna(axis=1, how='all')
        
        # Flag duplicate periods
        df['duplicate_period'] = df.duplicated(subset=['calendarYear', 'period'], keep=False)
        
        # Filter to FY only for annual period
        if period.lower() == 'annual':
            df = df[df['period'] == 'FY']
        
        # Calculate internal validation values
        # Current Assets
        df['totalCurrentAssets_internal'] = (
            df['cashAndCashEquivalents'].fillna(0) +
            df['shortTermInvestments'].fillna(0) +
            df['netReceivables'].fillna(0) +
            df['inventory'].fillna(0) +
            df['otherCurrentAssets'].fillna(0)
        )
        
        # Non-Current Assets
        df['totalNonCurrentAssets_internal'] = (
            df['propertyPlantEquipmentNet'].fillna(0) +
            df['goodwill'].fillna(0) +
            df['intangibleAssets'].fillna(0) +
            df['longTermInvestments'].fillna(0) +
            df['taxAssets'].fillna(0) +
            df['otherNonCurrentAssets'].fillna(0)
        )
        
        # Total Assets
        df['totalAssets_internal'] = (
            df['totalCurrentAssets'].fillna(0) +
            df['totalNonCurrentAssets'].fillna(0)
        )
        
        # Current Liabilities
        df['totalCurrentLiabilities_internal'] = (
            df['accountPayables'].fillna(0) +
            df['shortTermDebt'].fillna(0) +
            df['taxPayables'].fillna(0) +
            df['deferredRevenue'].fillna(0) +
            df['otherCurrentLiabilities'].fillna(0)
        )
        
        # Non-Current Liabilities
        df['totalNonCurrentLiabilities_internal'] = (
            df['longTermDebt'].fillna(0) +
            df['deferredRevenueNonCurrent'].fillna(0) +
            df['deferredTaxLiabilitiesNonCurrent'].fillna(0) +
            df['otherNonCurrentLiabilities'].fillna(0) +
            df['capitalLeaseObligations'].fillna(0)
        )
        
        # Total Liabilities
        df['totalLiabilities_internal'] = (
            df['totalCurrentLiabilities'].fillna(0) +
            df['totalNonCurrentLiabilities'].fillna(0)
        )
        
        # Stockholders Equity
        df['totalStockholdersEquity_internal'] = (
            df['preferredStock'].fillna(0) +
            df['commonStock'].fillna(0) +
            df['retainedEarnings'].fillna(0) +
            df['accumulatedOtherComprehensiveIncomeLoss'].fillna(0) +
            df['othertotalStockholdersEquity'].fillna(0) +
            df['minorityInterest'].fillna(0)
        )
        
        # Total Liabilities and Stockholders Equity
        df['totalLiabilitiesAndStockholdersEquity_internal'] = (
            df['totalLiabilities'].fillna(0) +
            df['totalStockholdersEquity'].fillna(0)
        )
        
        # Calculate validation deltas
        delta_mappings = {
            'totalCurrentAssets': 'totalCurrentAssets_internal',
            'totalNonCurrentAssets': 'totalNonCurrentAssets_internal',
            'totalAssets': 'totalAssets_internal',
            'totalCurrentLiabilities': 'totalCurrentLiabilities_internal',
            'totalNonCurrentLiabilities': 'totalNonCurrentLiabilities_internal',
            'totalLiabilities': 'totalLiabilities_internal',
            'totalStockholdersEquity': 'totalStockholdersEquity_internal',
            'totalLiabilitiesAndStockholdersEquity': 'totalLiabilitiesAndStockholdersEquity_internal'
        }
        
        for reported_col, calc_col in delta_mappings.items():
            delta_col = f'{reported_col}_delta_rpt_vs_calc'
            df[delta_col] = df[reported_col].fillna(0) - df[calc_col].fillna(0)
        
        # Accounting equation check (Assets = Liabilities + Equity)
        df['accounting_equation_delta'] = (
            df['totalAssets'].fillna(0) - 
            df['totalLiabilitiesAndStockholdersEquity'].fillna(0)
        )
        
        # Drop temporary internal calculation columns
        calc_cols = [col for col in df.columns if col.endswith('_internal')]
        df = df.drop(columns=calc_cols)
        
        # Define standard column order
        standard_order = [
            'reportedCurrency', 'calendarYear', 'period', 'duplicate_period',
            # Current Assets
            'cashAndCashEquivalents', 'shortTermInvestments', 'netReceivables',
            'inventory', 'otherCurrentAssets', 'totalCurrentAssets',
            # Non-Current Assets
            'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets',
            'longTermInvestments', 'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets',
            # Total Assets
            'totalAssets',
            # Current Liabilities
            'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue',
            'otherCurrentLiabilities', 'totalCurrentLiabilities',
            # Non-Current Liabilities
            'longTermDebt', 'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
            'otherNonCurrentLiabilities', 'capitalLeaseObligations', 'totalNonCurrentLiabilities',
            # Total Liabilities
            'totalLiabilities',
            # Stockholders Equity
            'preferredStock', 'commonStock', 'retainedEarnings',
            'accumulatedOtherComprehensiveIncomeLoss', 'othertotalStockholdersEquity',
            'minorityInterest', 'totalStockholdersEquity',
            # Total L&E
            'totalLiabilitiesAndStockholdersEquity',
            # Validation Deltas
            'totalCurrentAssets_delta_rpt_vs_calc', 'totalNonCurrentAssets_delta_rpt_vs_calc',
            'totalAssets_delta_rpt_vs_calc',
            'totalCurrentLiabilities_delta_rpt_vs_calc', 'totalNonCurrentLiabilities_delta_rpt_vs_calc',
            'totalLiabilities_delta_rpt_vs_calc',
            'totalStockholdersEquity_delta_rpt_vs_calc',
            'totalLiabilitiesAndStockholdersEquity_delta_rpt_vs_calc',
            'accounting_equation_delta',
            # Metadata
            'fillingDate'
        ]
        
        mandatory_cols = ['reportedCurrency', 'calendarYear', 'period']
        
        # Build final column list
        if facs is None:
            final_cols = mandatory_cols.copy()
            for col in standard_order:
                if col in df.columns and col not in final_cols:
                    final_cols.append(col)
        else:
            clean_facs = [f for f in facs if f in df.columns and f not in (mandatory_cols + ['date', 'symbol'])]
            final_cols = mandatory_cols.copy()
            for col in standard_order:
                if col in clean_facs and col not in final_cols:
                    final_cols.append(col)
            for col in clean_facs:
                if col not in final_cols:
                    final_cols.append(col)
        
        result_df = df[final_cols]
        result_df.attrs['symbol'] = sym
        result_df.attrs['metadata'] = metadata
        result_df.attrs['period'] = period.lower()
        
        # Save to markdown if requested
        if save_md:
            df_to_markdown_with_metadata(result_df, save_md)
        
        return result_df

    except Exception as e:
        print(f"Error in fmp_balts for {sym}: {e}")
        return pd.DataFrame()
#---------------------------------------------------------------------------

def fmp_baltsFmt(df, add_ratios=False, add_asset_composition=False, add_risk_metrics=False, filename='balance_sheet.html'):
    r"""
    Restored: Working Capital and RE/TA Risk Metrics.
    Features: Asset/Liability % Composition, Ratios, Dynamic Indentation, and Charting.

    **************Formats a Financial Modeling Prep (FMP:NASDAQ) balance sheet DataFrame into an interactive HTML report.

    This function transforms raw financial data into a professional HTML document. It handles 
    automatic scaling, applies financial hierarchy styling, and injects Chart.js for 
    interactive row-level trend analysis.

    Args:
        df (pd.DataFrame): Raw balance sheet data (metrics as columns, periods as rows).
        add_ratios (bool): If True, appends liquidity and leverage ratios.
        add_asset_composition (bool): If True, inserts "% of Total Assets" for major line items.
        add_risk_metrics (bool): If True, appends solvency and capital health metrics.
        filename (str): Output path for the HTML file.

    Returns:
        None: Writes file to disk and opens it in the default web browser.

    Notes:
        ### Scaling Logic
        The function determines the scale based on the first column's 'Total Assets':
        - If Total Assets $\ge 1,000,000,000$, values are divided by $1,000,000$ (Millions).
        - Otherwise, values are divided by $1,000$ (Thousands).

        ### Formulas & Metrics
        When optional flags are enabled, the following calculations are applied:

        **1. Asset Composition**
        - $\text{Line Item \%} = \left( \frac{\text{Line Item Value}}{\text{Total Assets}} \right) \times 100$

        **2. Ratios**
        - $\text{Current Ratio} = \frac{\text{Total Current Assets}}{\text{Total Current Liabilities}}$
        - $\text{Debt-to-Equity} = \frac{\text{Total Debt}}{\text{Total Stockholders Equity}}$

        **3. Risk Metrics**
        - $\text{Working Capital} = \text{Total Current Assets} - \text{Total Current Liabilities}$
        - $\text{TCE Ratio (Tangible Common Equity)} = \frac{\text{Equity} - \text{Goodwill} - \text{Intangibles}}{\text{Total Assets}} \times 100$
        - $\text{RE/TA} = \left( \frac{\text{Retained Earnings}}{\text{Total Assets}} \right) \times 100$
    """
    friendly_names = {
        'date': 'Date', 'reportedCurrency': 'Currency', 'calendarYear': 'Year', 'period': 'Quarter',
        'cashAndCashEquivalents': 'Cash & Equivalents', 'shortTermInvestments': 'Short-term Investments',
        'netReceivables': 'Net Receivables', 'inventory': 'Inventory', 'otherCurrentAssets': 'Other Current Assets',
        'totalCurrentAssets': 'Total Current Assets', 'propertyPlantEquipmentNet': 'PP&E (Net)',
        'goodwill': 'Goodwill', 'intangibleAssets': 'Intangible Assets', 'longTermInvestments': 'Long-term Investments',
        'taxAssets': 'Tax Assets', 'otherNonCurrentAssets': 'Other Non-Current Assets',
        'totalNonCurrentAssets': 'Total Non-Current Assets', 'totalAssets': 'Total Assets',
        'accountPayables': 'Accounts Payable', 'shortTermDebt': 'Short-term Debt',
        'taxPayables': 'Tax Payables', 'deferredRevenue': 'Deferred Revenue',
        'otherCurrentLiabilities': 'Other Current Liabilities', 'totalCurrentLiabilities': 'Total Current Liabilities',
        'longTermDebt': 'Long-term Debt', 'capitalLeaseObligations': 'Capital Lease Obligations',
        'deferredTaxLiabilitiesNonCurrent': 'Deferred Tax Liabilities', 'otherNonCurrentLiabilities': 'Other Non-Current Liabilities',
        'totalNonCurrentLiabilities': 'Total Non-Current Liabilities', 'totalLiabilities': 'Total Liabilities',
        'commonStock': 'Common Stock', 'retainedEarnings': 'Retained Earnings',
        'accumulatedOtherComprehensiveIncomeLoss': 'Accumulated Other Comprehensive Income',
        'totalStockholdersEquity': 'Shareholders\' Equity', 'totalLiabilitiesAndTotalEquity': 'Total Liabilities & Equity',
        'totalInvestments': 'Total Investments', 'totalDebt': 'Total Debt', 'netDebt': 'Net Debt'
    }

    df_work = df.copy().T
    df_work = df_work.drop(['fillingDate', 'acceptedDate'], errors='ignore')
    
    # Determine Scale based on Total Assets
    total_assets_val = pd.to_numeric(df_work.loc['totalAssets', df_work.columns[0]], errors='coerce')
    scale_factor = 1_000_000 if total_assets_val >= 1_000_000_000 else 1_000
    scale_label = 'Millions' if scale_factor == 1_000_000 else 'Thousands'

    # 1. ADD COMPOSITION (%) - Assets & Liabilities
    if add_asset_composition:
        comp_tags = [
            'cashAndCashEquivalents', 'shortTermInvestments', 'netReceivables', 'inventory', 
            'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 
            'goodwill', 'intangibleAssets', 'longTermInvestments', 'taxAssets', 
            'otherNonCurrentAssets', 'totalNonCurrentAssets',
            'accountPayables', 'deferredRevenue', 'totalDebt', 'retainedEarnings'
        ]
        
        rows_to_insert = []
        for tag in comp_tags:
            if tag in df.columns:
                pct_row = pd.Series(index=df_work.columns, name=f"{tag}_pct", dtype=float)
                for col in df_work.columns:
                    val = pd.to_numeric(df.loc[df.index[df.index.get_loc(col)], tag], errors='coerce')
                    total = pd.to_numeric(df.loc[df.index[df.index.get_loc(col)], 'totalAssets'], errors='coerce')
                    pct_row[col] = (val / total * 100) if total != 0 else np.nan
                rows_to_insert.append((tag, pct_row))

        for parent_tag, pct_row in reversed(rows_to_insert):
            if parent_tag in df_work.index:
                idx_pos = df_work.index.get_loc(parent_tag)
                df_top = df_work.iloc[:idx_pos+1]
                df_bottom = df_work.iloc[idx_pos+1:]
                pct_row.name = f"% of Total Assets ({friendly_names.get(parent_tag, parent_tag)})"
                df_work = pd.concat([df_top, pd.DataFrame([pct_row]), df_bottom])

    # 2. ADD RISK METRICS (Restored Working Capital and RE/TA)
    if add_risk_metrics:
        # Working Capital (Scaled)
        wc = (pd.to_numeric(df['totalCurrentAssets']) - pd.to_numeric(df['totalCurrentLiabilities'])) / scale_factor
        df_work = pd.concat([df_work, pd.DataFrame([pd.Series(wc.values, index=df_work.columns, name='Working Capital')])])
        
        # TCE Ratio (%)
        tce = ((pd.to_numeric(df['totalStockholdersEquity']) - pd.to_numeric(df['goodwill']) - pd.to_numeric(df['intangibleAssets'])) / pd.to_numeric(df['totalAssets'])) * 100
        df_work = pd.concat([df_work, pd.DataFrame([pd.Series(tce.values, index=df_work.columns, name='TCE Ratio (%)')])])

        # RE/TA (%) - Retained Earnings / Total Assets
        reta = (pd.to_numeric(df['retainedEarnings']) / pd.to_numeric(df['totalAssets'])) * 100
        df_work = pd.concat([df_work, pd.DataFrame([pd.Series(reta.values, index=df_work.columns, name='RE/TA (%)')])])

    # 3. ADD RATIOS
    if add_ratios:
        curr_ratio = pd.to_numeric(df['totalCurrentAssets']) / pd.to_numeric(df['totalCurrentLiabilities'])
        df_work = pd.concat([df_work, pd.DataFrame([pd.Series(curr_ratio.values, index=df_work.columns, name='Current Ratio')])])
        dte = pd.to_numeric(df['totalDebt']) / pd.to_numeric(df['totalStockholdersEquity'])
        df_work = pd.concat([df_work, pd.DataFrame([pd.Series(dte.values, index=df_work.columns, name='Debt-to-Equity')])])

    # Initial Scaling for main numeric rows
    meta_rows = ['date', 'reportedCurrency', 'calendarYear', 'period']
    for row in df_work.index:
        if row not in meta_rows and "%" not in str(row) and "Ratio" not in str(row) and row != 'Working Capital':
            for col in df_work.columns:
                val = pd.to_numeric(df_work.loc[row, col], errors='coerce')
                if pd.notna(val) and row in friendly_names:
                    df_work.loc[row, col] = val / scale_factor

    # Apply UI Hierarchy
    grand_totals = ['Total Assets', 'Total Liabilities & Equity', 'Total Debt', 'Net Debt']
    subtotals = ['Total Current Assets', 'Total Non-Current Assets', 'Total Current Liabilities', 'Total Non-Current Liabilities', 'Total Liabilities', "Shareholders' Equity"]

    def apply_styles(label):
        clean = friendly_names.get(label, label)
        if clean in grand_totals: return f"<strong>{clean}</strong>"
        if clean in subtotals: return f"&nbsp;&nbsp;{clean}"
        if "%" in str(label): return f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>{label}</em>"
        return f"&nbsp;&nbsp;&nbsp;&nbsp;{clean}"

    df_work.index = [apply_styles(idx) for idx in df_work.index]

    # Cell Value Formatting
    for idx in df_work.index:
        for col in df_work.columns:
            val = df_work.loc[idx, col]
            if isinstance(val, (int, float)) and pd.notna(val):
                fmt = "{:.2f}" if ("%" in str(idx) or "Ratio" in str(idx)) else "{:,.2f}"
                df_work.loc[idx, col] = fmt.format(val)

    # HTML Output with Charting Logic
    table_html = df_work.to_html(classes='balance-sheet', border=0, escape=False)
    
    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, system-ui, sans-serif; padding: 40px; background-color: #f0f2f5; }}
        .container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h2 {{ color: #1a1a1a; border-bottom: 3px solid #4CAF50; padding-bottom: 15px; margin-bottom: 20px; }}
        .balance-sheet {{ border-collapse: collapse; width: 100%; font-size: 13px; color: #333; }}
        .balance-sheet thead th {{ background-color: #4CAF50; color: white; padding: 12px 10px; text-align: right; text-transform: uppercase; letter-spacing: 1px; font-size: 11px; }}
        .balance-sheet tbody th {{ text-align: left !important; padding: 8px 15px; border: 1px solid #e0e0e0; border-right: 2px solid #ccc; white-space: nowrap; cursor: pointer; font-family: "SF Mono", monospace; background-color: #fff; }}
        .balance-sheet tbody th:hover {{ background-color: #e8f5e9; color: #2e7d32; }}
        .balance-sheet tbody td {{ border: 1px solid #e0e0e0; padding: 8px; text-align: right; }}
        .balance-sheet tbody tr:nth-child(even) {{ background-color: #fafafa; }}
        strong {{ font-weight: 800 !important; color: #000; }}
        em {{ color: #777; font-size: 0.85em; }}
        #chartModal {{ display: none; position: fixed; z-index: 100; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); }}
        .modal-content {{ background: white; margin: 5% auto; padding: 20px; border-radius: 12px; width: 80%; max-width: 900px; }}
        .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container"><h2>Balance Sheet (in {scale_label})</h2><div style="overflow-x: auto;">{table_html}</div></div>
    <div id="chartModal"><div class="modal-content"><span class="close" onclick="closeModal()">&times;</span><canvas id="rowChart"></canvas></div></div>
    <script>
        let myChart = null;
        document.querySelectorAll(".balance-sheet tbody th").forEach(header => {{
            header.onclick = function() {{
                const row = this.parentElement;
                const label = this.innerText.trim();
                const labels = Array.from(document.querySelectorAll(".balance-sheet thead th")).slice(1).map(th => th.innerText);
                const data = Array.from(row.querySelectorAll("td")).map(td => parseFloat(td.innerText.replace(/,/g, '')));
                document.getElementById("chartModal").style.display = "block";
                if (myChart) myChart.destroy();
                myChart = new Chart(document.getElementById('rowChart'), {{
                    type: 'line', data: {{ labels, datasets: [{{ label, data, borderColor: '#4CAF50', backgroundColor: 'rgba(76,175,80,0.1)', fill: true, tension: 0.3 }}] }}
                }});
            }};
        }});
        function closeModal() {{ document.getElementById("chartModal").style.display = "none"; }}
    </script>
</body>
</html>
"""
    with open(filename, 'w', encoding='utf-8') as f: f.write(styled_html)
    webbrowser.open('file://' + os.path.abspath(filename))

#----------------------------------------------------------------------------
#MOD  011826 4:41 PM
def fmp_baltsC(sym, facs=None, period='quarter', limit=400,
              qc_debt=True,
              qc_abs_tol=5_000_000,        # $5m
              qc_rel_tol=0.05,             # 5%
              qc_max_date_gap_days=45,     # AR->STD alignment guard
              qc_override_direction="up",  # "up" (recommended), "both"
              qc_return_cols=True):
    '''
    Retrieves standardized Balance Sheet data from the FMP v3 endpoint.
    Adds Smart Correction (equity_error_delta), auto-pruning of empty columns,
    AND a debt QC/override layer using fmp_baltsar() to mitigate known standardized debt issues.

    Debt QC philosophy (conservative):
    - Default debt is standardized EX-LEASES: shortTermDebt + longTermDebt
    - Compute AR debt from as-reported debt COMPONENTS (not aggregates) when coverage is "strong"
    - Override only when AR indicates standardized debt is UNDERSTATED by a large margin
      (this catches MLAB-type failures without wrecking the universe)

    Parameters
    ----------
    qc_debt : bool
        If True, compute debt_used_ex_leases with AR-based upward override when warranted.
    qc_override_direction : {"up","both"}
        "up": only override when AR debt > STD debt by thresholds (recommended)
        "both": allow overrides in either direction when AR is strong (riskier)

    Output columns added when qc_debt=True
    ------------------------------------
    debt_std_ex_leases
    debt_ar_ex_leases
    debt_ar_tag_count
    debt_gap_std_minus_ar
    debt_used_ex_leases
    debt_source
    debt_qc_flag
    cash_used
    net_debt_used_ex_leases
    ar_debt_date_aligned

    Notes
    -----
    - This function does NOT compute EV; it produces a better debt input for EV.
    - You can still request specific facs; QC columns are appended if qc_return_cols=True.
    '''

    # -----------------------------
    # 0) Standardized fetch (same as your original)
    # -----------------------------
    available_tags = [
        'fillingDate', 'acceptedDate', 'cashAndCashEquivalents', 'shortTermInvestments',
        'cashAndShortTermInvestments', 'netReceivables', 'inventory', 'otherCurrentAssets',
        'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill', 'intangibleAssets',
        'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets',
        'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets',
        'accountPayables', 'shortTermDebt', 'taxPayables', 'deferredRevenue',
        'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt',
        'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent',
        'otherNonCurrentLiabilities', 'totalNonCurrentLiabilities', 'otherLiabilities',
        'capitalLeaseObligations', 'totalLiabilities', 'preferredStock', 'commonStock',
        'retainedEarnings', 'accumulatedOtherComprehensiveIncomeLoss',
        'othertotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity',
        'totalLiabilitiesAndStockholdersEquity', 'minorityInterest',
        'totalLiabilitiesAndTotalEquity', 'totalInvestments', 'totalDebt', 'netDebt'
    ]

    sym = sym.upper().strip()
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}?period={period}&limit={limit}&apikey={apikey}"
    url = requote_uri(url)

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))
        if not stuff or not isinstance(stuff, list):
            return pd.DataFrame()

        df = pd.DataFrame(stuff)

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = 'date'

        df = df.sort_index()
        df = df.dropna(axis=1, how='all')

        # Smart Correction
        if all(c in df.columns for c in ['totalAssets','totalLiabilities','totalStockholdersEquity']):
            df['equity_error_delta'] = (
                df['totalAssets'].fillna(0)
                - (df['totalLiabilities'].fillna(0) + df['totalStockholdersEquity'].fillna(0))
            )

        mandatory_cols = ['reportedCurrency', 'calendarYear', 'period']

        if facs is None:
            facs = available_tags

        clean_facs = [f for f in facs if f in df.columns and f not in (mandatory_cols + ['date','symbol'])]
        final_cols = mandatory_cols + clean_facs
        out = df[final_cols].copy()

        # -----------------------------
        # 1) Debt QC / override layer (optional)
        # -----------------------------
        if not qc_debt:
            return out

        # Ensure numeric types where needed
        for c in ['shortTermDebt','longTermDebt','capitalLeaseObligations','totalDebt','netDebt',
                  'cashAndShortTermInvestments','cashAndCashEquivalents','shortTermInvestments']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Standardized debt (ex leases)
        debt_std = df.get('shortTermDebt', np.nan) + df.get('longTermDebt', np.nan)

        # Cash used (cash + ST inv)
        cash_used = df.get('cashAndShortTermInvestments', np.nan)
        if isinstance(cash_used, pd.Series):
            cash_used = cash_used.fillna(df.get('cashAndCashEquivalents', np.nan) + df.get('shortTermInvestments', 0.0))

        # Defaults
        debt_used = debt_std.copy()
        debt_source = pd.Series(["standardized"] * len(df), index=df.index)
        debt_qc_flag = pd.Series([False] * len(df), index=df.index)
        debt_ar = pd.Series([np.nan] * len(df), index=df.index)
        debt_ar_tag_count = pd.Series([0] * len(df), index=df.index)
        ar_date_aligned = pd.Series([pd.NaT] * len(df), index=df.index)

        # --- As-reported debt component extraction ---
        # Only use AR components (not aggregates) to avoid lease/double-count ambiguity
        AR_DEBT_COMPONENT_TAGS = [
            # common debt components across schemas
            'debtcurrent',
            'longtermdebtnoncurrent',
            'longtermdebtcurrent',
            'shorttermborrowings',
            'shorttermdebt',
            'commercialpaper',
            'lineofcredit',
            'linesofcreditcurrent',
            'longtermlineofcredit',
            'secureddebtcurrent',
            'securedlongtermdebt',
            'notespayablecurrent',
            'longtermnotespayable',
            'othershorttermborrowings',
            'convertibledebtcurrent',
            'convertibledebtnoncurrent'
        ]

        # Fetch AR once (same period/limit)
        ar = fmp_baltsar(sym, facs=None, period=period, limit=limit)
        if ar is not None and not ar.empty:
            ar = ar.copy().sort_index()

            # Compute AR debt component sum per AR date
            # (only sum tags that exist and are numeric)
            ar_debt_cols = [c for c in AR_DEBT_COMPONENT_TAGS if c in ar.columns]
            if ar_debt_cols:
                ar_num = ar[ar_debt_cols].apply(pd.to_numeric, errors='coerce')
                ar_debt_sum = ar_num.fillna(0).sum(axis=1)
                ar_tag_n = (ar_num.notna() & (ar_num != 0)).sum(axis=1)

                # Align AR to STD dates (asof backward), with date gap guard
                std_dates = df.index.to_series().rename("std_date").to_frame()
                ar_tmp = pd.DataFrame({
                    "ar_date": ar_debt_sum.index,
                    "ar_debt": ar_debt_sum.values,
                    "ar_tag_n": ar_tag_n.values
                }).sort_values("ar_date")

                std_tmp = std_dates.reset_index(drop=True).sort_values("std_date")

                aligned = pd.merge_asof(
                    std_tmp,
                    ar_tmp,
                    left_on="std_date",
                    right_on="ar_date",
                    direction="backward",
                    allow_exact_matches=True
                )

                # Push aligned values back onto index
                aligned.index = df.index
                debt_ar = aligned["ar_debt"]
                debt_ar_tag_count = aligned["ar_tag_n"]
                ar_date_aligned = aligned["ar_date"]

                # Date-gap guard
                # Date-gap guard
                ar_dt = pd.to_datetime(ar_date_aligned, errors="coerce")
                gap_days = (pd.Series(df.index, index=df.index) - ar_dt).abs().dt.days
                gap_ok = gap_days <= qc_max_date_gap_days

                gap_ok = gap_days <= qc_max_date_gap_days

                # AR "strong" coverage rule (conservative)
                # require >=2 nonzero component tags
                ar_strong = (debt_ar_tag_count >= 2) & gap_ok & debt_ar.notna() & (debt_ar > 0)

                # Compute mismatch
                debt_gap = debt_std - debt_ar
                rel_gap = (debt_gap.abs() / debt_ar.abs()).replace([np.inf, -np.inf], np.nan)

                # Override rule
                if qc_override_direction.lower() == "up":
                    override = ar_strong & (debt_ar > debt_std) & (debt_gap.abs() > qc_abs_tol) & (rel_gap > qc_rel_tol)
                else:
                    override = ar_strong & (debt_gap.abs() > qc_abs_tol) & (rel_gap > qc_rel_tol)

                debt_used = debt_used.where(~override, debt_ar)
                debt_source = debt_source.where(~override, "as_reported_override")
                debt_qc_flag = debt_qc_flag.where(~override, True)

                # Keep debt_gap for reporting (std - ar)
                debt_gap_std_minus_ar = debt_gap
            else:
                debt_gap_std_minus_ar = pd.Series([np.nan] * len(df), index=df.index)
        else:
            debt_gap_std_minus_ar = pd.Series([np.nan] * len(df), index=df.index)

        net_debt_used = debt_used - cash_used

        # Append QC columns (don’t break existing selection logic)
        qc_cols = pd.DataFrame({
            "debt_std_ex_leases": debt_std,
            "debt_ar_ex_leases": debt_ar,
            "debt_ar_tag_count": debt_ar_tag_count,
            "debt_gap_std_minus_ar": debt_gap_std_minus_ar,
            "debt_used_ex_leases": debt_used,
            "debt_source": debt_source,
            "debt_qc_flag": debt_qc_flag,
            "cash_used": cash_used,
            "net_debt_used_ex_leases": net_debt_used,
            "ar_debt_date_aligned": ar_date_aligned
        }, index=df.index)

        if qc_return_cols:
            out = out.join(qc_cols, how="left")

        return out

    except Exception as e:
        print(f"Error in fmp_balts for {sym}: {e}")
        return pd.DataFrame()


#--------------------------------------------------------------------
## MOD  01/18/2026 7:00 AM

def fmp_baltsar(sym, facs=None, period='quarter', limit=400):
    '''As-Reported Balance Sheet with Discovery-focused Pruning.'''
    sym = sym.upper().strip()
    p = "annual" if period.lower() in ("year", "annual") else "quarter"
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement-as-reported/{sym}?period={p}&limit={int(limit)}&apikey={apikey}"
    url = requote_uri(url)

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))
        if not stuff: return pd.DataFrame()

        df = pd.DataFrame(stuff)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = "date"
        
        df = df.sort_index()

        # DROP strictly empty columns (improves scannability for AR data)
        df = df.dropna(axis=1, how='all')

        # Smart Correction for AR (Mapping verification)
        if all(col in df.columns for col in ['assets', 'liabilities', 'stockholdersequity']):
            df['equity_error_delta'] = (df['assets'].fillna(0) - 
                                       (df['liabilities'].fillna(0) + 
                                        df['stockholdersequity'].fillna(0)))

        if facs:
            clean_facs = [f for f in facs if f != 'date']
            df = df.reindex(columns=clean_facs)
            
        return df

    except Exception as e:
        print(f"Error in fmp_baltsar for {sym}: {e}")
        return pd.DataFrame()
#----------------------------------------------------------------------------
#MOD 02/02/2026  8:25 AM

def fmp_balarFmt(df, asset_tag='assets'):
    """
    Format as-reported balance sheet with K/M/B scaling.
    Keeps all original tag names unchanged.
    
    Parameters:
    -----------
    df : DataFrame
        Raw output from fmp_baltsar() with dates in rows, tags in columns
    asset_tag : str
        Tag name to use for determining scale (default 'assets')
        Can be 'assets', 'totalassets', or any column name in the data
    
    Returns:
    --------
    DataFrame with scaled values formatted with commas and 2 decimals
    Also prints title indicating scale (K/M/B)
    
    Example:
    --------
    >>> raw_df = fmp_baltsar('F', period='quarter', limit=12)
    >>> formatted = format_as_reported(raw_df, asset_tag='assets')
    """
    
    # Make a copy and transpose (tags to index, dates to columns)
    df_work = df.copy()
    df_work = df_work.T
    
    # Find the asset tag (case-insensitive search)
    asset_tag_lower = asset_tag.lower()
    matching_tags = [idx for idx in df_work.index if asset_tag_lower in str(idx).lower()]
    
    if not matching_tags:
        raise ValueError(f"Asset tag '{asset_tag}' not found in data. Available tags: {list(df_work.index)}")
    
    # Use the first matching tag
    found_asset_tag = matching_tags[0]
    
    # Use the first date column's asset value for scaling decision
    first_col = df_work.columns[0]
    asset_value = df_work.loc[found_asset_tag, first_col]
    
    # Determine scale based on asset value
    if asset_value < 1_000_000:
        scale_factor = 1
        scale_label = 'Units'
    elif asset_value < 1_000_000_000:
        scale_factor = 1_000
        scale_label = 'Thousands'
    else:
        scale_factor = 1_000_000
        scale_label = 'Millions'
    
    # Identify numeric rows (exclude date/period/symbol type rows)
    non_numeric_rows = ['date', 'symbol', 'period', 'reportedCurrency', 'fillingDate', 'acceptedDate']
    numeric_rows = [idx for idx in df_work.index if idx not in non_numeric_rows]
    
    # Scale numeric values
    for col in df_work.columns:
        for row in numeric_rows:
            if row in df_work.index and pd.notna(df_work.loc[row, col]):
                try:
                    df_work.loc[row, col] = df_work.loc[row, col] / scale_factor
                except (TypeError, ValueError):
                    # Skip non-numeric values
                    pass
    
    # Format numeric values with commas and 2 decimals
    for idx in numeric_rows:
        if idx in df_work.index:
            for col in df_work.columns:
                val = df_work.loc[idx, col]
                if isinstance(val, (int, float)) and pd.notna(val):
                    df_work.loc[idx, col] = f"{val:,.2f}"
    
    # Print title with scale
    print(f"Balance Sheet (in {scale_label})")
    print()
    
    return df_work

#-----------------------------------------------------------------------------
# 2026-02-07 04:57:22
def fmp_incts(sym, period='quarter', limit=8, facs=None, save_md=None):
    '''
    Retrieves standardized Income Statement data from the FMP v3 endpoint.
    Displays FMP reported totals and validates against calculated component sums.
    
    Parameters:
    -----------
    sym : str
        The stock ticker symbol (e.g., 'AAPL').
    
    period : str, default='quarter'
        The reporting period to retrieve:
        - 'quarter' : Quarterly financial statements (default)
        - 'annual'  : Annual financial statements (FY only)
        - 'ttm'     : Trailing Twelve Months (rolling 4-quarter sum)
    
    limit : int, default=400
        Maximum number of periods to retrieve from FMP.
        - For 'quarter' and 'annual': Returns up to `limit` periods
        - For 'ttm': Returns approximately `limit - 3` TTM periods
          (Formula: ttm_periods = limit - 3, so limit = desired_ttm_periods + 3)
          Examples: limit=8 → 5 TTM periods, limit=23 → 20 TTM periods
    
    facs : list, optional
        List of specific line items to return. If None, returns the full standard set.
        
        Mandatory Columns (Always Returned):
        ['reportedCurrency', 'calendarYear', 'period']
        
        AVAILABLE TAGS:
        Components:
        ['revenue', 'costOfRevenue', 'researchAndDevelopmentExpenses', 
         'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses', 
         'otherExpenses', 'depreciationAndAmortization', 'interestIncome', 
         'interestExpense', 'totalOtherIncomeExpensesNet', 'incomeTaxExpense',
         'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil']
        
        FMP Reported Totals (from 10-Q/10-K):
        ['grossProfit', 'sga', 'operatingExpenses', 'operatingIncome', 'ebitda',
         'incomeBeforeTax', 'netIncome']
        
        Validation Deltas (Reported - Calculated from Components):
        ['grossProfit_delta_rpt_vs_calc', 'sga_delta_rpt_vs_calc', 'operatingExpenses_delta_rpt_vs_calc',
         'operatingIncome_delta_rpt_vs_calc', 'ebitda_delta_rpt_vs_calc', 'netIncome_delta_rpt_vs_calc']
        
        Data Quality Flags:
        ['duplicate_period'] - True if multiple filings exist for same calendarYear/period
        
        Metadata:
        ['fillingDate', 'acceptedDate', 'link', 'finalLink', 'cik']
    
    save_md : str, optional
        Filepath to save DataFrame with metadata as markdown file.
        If provided, saves output with company metadata header.
        Example: save_md='azenta_financials.md'
    
    Returns:
    --------
    pd.DataFrame
        Income statement data with date index and symbol stored in df.attrs['symbol']
        Company metadata stored in df.attrs['metadata'] containing:
        ['companyName', 'cik', 'isin', 'cusip', 'currency', 'exchangeShortName',
         'industry', 'sector', 'country']
    
    LLM Context for Validation Columns:
    ------------------------------------
    This dataset includes validation columns (suffix: _delta_rpt_vs_calc) that verify 
    financial statement internal consistency.
    
    Formula: Δ = FMP_reported_value - calculated_from_components
    
    Validation Columns & Their Formulas:
    - grossProfit_delta_rpt_vs_calc = grossProfit(rpt) - (revenue - costOfRevenue)
    - sga_delta_rpt_vs_calc = sga(rpt) - (generalAndAdministrativeExpenses + sellingAndMarketingExpenses)
    - operatingExpenses_delta_rpt_vs_calc = operatingExpenses(rpt) - (researchAndDevelopmentExpenses + sga + otherExpenses)
    - operatingIncome_delta_rpt_vs_calc = operatingIncome(rpt) - (grossProfit - operatingExpenses)
    - ebitda_delta_rpt_vs_calc = ebitda(rpt) - (operatingIncome + depreciationAndAmortization)
    - netIncome_delta_rpt_vs_calc = netIncome(rpt) - (incomeBeforeTax - incomeTaxExpense)
    
    Interpreting Validation Deltas:
    Δ = 0        → Components reconcile perfectly with reported total ✓
    Δ > 0        → Reported total exceeds component sum
                   CAUSE: Component breakdown not disclosed in filing
                   EXAMPLE: Company reports SG&A as single line without G&A/S&M split
    Δ < 0        → Component sum exceeds reported total
                   CAUSE: Potential data quality issue or parsing error 🚩
    Small Δ      → Rounding differences (typically <0.1% of revenue)
    
    Common Patterns:
    1. sga_delta_rpt_vs_calc = sga amount, with G&A=0 and S&M=0
       → Company doesn't break out SG&A components (normal for many companies)
    
    2. operatingExpenses_delta_rpt_vs_calc = same as sga_delta_rpt_vs_calc
       → OpEx delta inherits the SG&A breakdown issue
    
    3. All deltas = 0 except small ebitda_delta_rpt_vs_calc
       → Likely D&A timing differences or rounding
    
    4. Unexpected large delta (>1% of line item)
       → Flag for manual review of source filing
    
    When asked to verify data quality:
    1. Check if delta = 0 (perfect reconciliation)
    2. If delta > 0, check if components exist (look for zeros)
    3. If components are zero, delta indicates missing breakdown (expected)
    4. If components exist but don't sum, flag as data quality issue
    5. Calculate delta as % of reported value to assess materiality
    
    Data Quality Flag - duplicate_period:
    - False: Single filing for this period (normal)
    - True: Multiple filings exist for same calendarYear/period combination
           CAUSES: Amended filings, restatements, or FMP data quality issues
           ACTION: Review fillingDate to identify most recent filing, or 
                   manually verify which filing to use
    
    TTM Period Notes:
    - Financial line items are summed over rolling 4 quarters
    - EPS metrics are calculated as: TTM Net Income / Average TTM Diluted Shares
    - Share counts are averaged (not summed) over the 4-quarter period
    - duplicate_period flag not applicable (TTM aggregates across periods)
    
    Metadata Notes:
    - Company metadata is fetched via fmp_profF() and attached to df.attrs['metadata']
    - Metadata persists in Python session but not in all file formats
    - Use save_md parameter to export with metadata preserved in markdown format
    - Access metadata: df.attrs['metadata']
        
    Examples:
    ---------
    >>> # Get last 8 quarters
    >>> df = fmp_incts('AAPL', period='quarter', limit=8)
    
    >>> # Get last 5 years annual
    >>> df = fmp_incts('AAPL', period='annual', limit=5)
    
    >>> # Get last 20 TTM periods
    >>> df = fmp_incts('AAPL', period='ttm', limit=23)
    
    >>> # Get specific fields only
    >>> df = fmp_incts('AAPL', facs=['revenue', 'netIncome', 'eps'])
    
    >>> # Check for duplicate filings
    >>> df = fmp_incts('AZTA', period='quarter', limit=8)
    >>> if df['duplicate_period'].any():
    >>>     print("Warning: Multiple filings detected")
    >>>     print(df[df['duplicate_period']][['calendarYear', 'period', 'fillingDate']])
    
    >>> # Access company metadata
    >>> df = fmp_incts('AZTA')
    >>> print(df.attrs['metadata'])
    
    >>> # Save with metadata to markdown
    >>> df = fmp_incts('AZTA', save_md='azenta_financials.md')
    '''
    
    sym = sym.upper().strip()
    
    # Fetch company profile metadata
    try:
        profile = fmp_profF(sym)
        metadata = {
            'companyName': profile.get('companyName'),
            'cik': profile.get('cik'),
            'isin': profile.get('isin'),
            'cusip': profile.get('cusip'),
            'currency': profile.get('currency'),
            'exchangeShortName': profile.get('exchangeShortName'),
            'industry': profile.get('industry'),
            'sector': profile.get('sector'),
            'country': profile.get('country')
        }
    except Exception as e:
        print(f"Warning: Could not fetch metadata for {sym}: {e}")
        metadata = {}
    
    fetch_period = 'quarter' if period.lower() == 'ttm' else period
    
    # URL construction
    url = f'https://financialmodelingprep.com/api/v3/income-statement/{sym}?period={fetch_period}&limit={limit}&apikey={apikey}'

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))
        if not stuff or not isinstance(stuff, list): 
            return pd.DataFrame()

        df = pd.DataFrame(stuff)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = 'date'
        
        df = df.sort_index()

        if 'sellingGeneralAndAdministrativeExpenses' in df.columns:
            df = df.rename(columns={'sellingGeneralAndAdministrativeExpenses': 'sga'})

        df = df.dropna(axis=1, how='all')

        # Flag duplicate periods (for quarter and annual only, not TTM)
        if period.lower() in ['quarter', 'annual']:
            df['duplicate_period'] = df.duplicated(subset=['calendarYear', 'period'], keep=False)

        # TTM period calculations
        if period.lower() == 'ttm':
            df = df.reset_index()
            window_size = df['period'].nunique() if df['period'].nunique() in [2, 4] else 4
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            cols_to_sum = [c for c in numeric_cols if 'eps' not in c.lower() and 'Year' not in c and 'shs' not in c.lower()]

            # FIRST: Average share counts on original data
            if 'weightedAverageShsOutDil' in df.columns:
                df['weightedAverageShsOutDil'] = df['weightedAverageShsOutDil'].rolling(window=window_size).mean()
            
            if 'weightedAverageShsOut' in df.columns:
                df['weightedAverageShsOut'] = df['weightedAverageShsOut'].rolling(window=window_size).mean()
            
            # THEN: Sum financial statement line items
            df[cols_to_sum] = df[cols_to_sum].rolling(window=window_size).sum()
            
            # Drop incomplete rolling windows
            df = df.dropna(subset=[cols_to_sum[0]]).set_index('date')
            
            # Calculate TTM EPS after dropna (when we have valid TTM periods)
            if 'netIncome' in df.columns and 'weightedAverageShsOutDil' in df.columns:
                df['epsdiluted'] = df['netIncome'] / df['weightedAverageShsOutDil']
            
            if 'netIncome' in df.columns and 'weightedAverageShsOut' in df.columns:
                df['eps'] = df['netIncome'] / df['weightedAverageShsOut']

        elif period.lower() == 'annual':
            df = df[df['period'] == 'FY']

        # Calculate internal validation values
        df['grossProfit_internal'] = df['revenue'].fillna(0) - df['costOfRevenue'].fillna(0)
        
        df['sga_internal'] = (df['generalAndAdministrativeExpenses'].fillna(0) + 
                              df['sellingAndMarketingExpenses'].fillna(0))
        
        df['operatingExpenses_internal'] = (df['researchAndDevelopmentExpenses'].fillna(0) + 
                                            df['sga_internal'].fillna(0) + 
                                            df['otherExpenses'].fillna(0))
        
        df['operatingIncome_internal'] = df['grossProfit'].fillna(0) - df['operatingExpenses'].fillna(0)
        
        df['ebitda_internal'] = df['operatingIncome'].fillna(0) + df['depreciationAndAmortization'].fillna(0)
        
        df['netIncome_internal'] = df['incomeBeforeTax'].fillna(0) - df['incomeTaxExpense'].fillna(0)
        
        # Calculate validation deltas using loop
        delta_mappings = {
            'grossProfit': 'grossProfit_internal',
            'sga': 'sga_internal',
            'operatingExpenses': 'operatingExpenses_internal',
            'operatingIncome': 'operatingIncome_internal',
            'ebitda': 'ebitda_internal',
            'netIncome': 'netIncome_internal'
        }
        
        for reported_col, calc_col in delta_mappings.items():
            delta_col = f'{reported_col}_delta_rpt_vs_calc'
            df[delta_col] = df[reported_col].fillna(0) - df[calc_col].fillna(0)

        # Drop temporary internal calculation columns
        calc_cols = [col for col in df.columns if col.endswith('_internal')]
        df = df.drop(columns=calc_cols)

        # Define standard column order
        standard_order = [
            'reportedCurrency', 'calendarYear', 'period', 'duplicate_period',
            'revenue', 'costOfRevenue', 'grossProfit',
            'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
            'sellingAndMarketingExpenses', 'sga', 'otherExpenses', 'operatingExpenses',
            'depreciationAndAmortization', 'operatingIncome', 'ebitda',
            'totalOtherIncomeExpensesNet', 'interestIncome', 'interestExpense',
            'incomeBeforeTax', 'incomeTaxExpense', 'netIncome',
            'eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil',
            'grossProfit_delta_rpt_vs_calc', 'sga_delta_rpt_vs_calc', 'operatingExpenses_delta_rpt_vs_calc',
            'operatingIncome_delta_rpt_vs_calc', 'ebitda_delta_rpt_vs_calc', 'netIncome_delta_rpt_vs_calc',
            'fillingDate'
        ]

        mandatory_cols = ['reportedCurrency', 'calendarYear', 'period']
        
        # Build final column list
        if facs is None:
            final_cols = mandatory_cols.copy()
            for col in standard_order:
                if col in df.columns and col not in final_cols:
                    final_cols.append(col)
        else:
            clean_facs = [f for f in facs if f in df.columns and f not in (mandatory_cols + ['date', 'symbol'])]
            final_cols = mandatory_cols.copy()
            for col in standard_order:
                if col in clean_facs and col not in final_cols:
                    final_cols.append(col)
            for col in clean_facs:
                if col not in final_cols:
                    final_cols.append(col)

        result_df = df[final_cols]
        result_df.attrs['symbol'] = sym
        result_df.attrs['metadata'] = metadata
        result_df.attrs['period'] = period.lower()
        # Save to markdown if requested
        if save_md:
            df_to_markdown_with_metadata(result_df, save_md)
        
        return result_df

    except Exception as e:
        print(f"Error in fmp_incts for {sym}: {e}")
        return pd.DataFrame()

#-----------------------------------------------------------------------
###Helper function for fmp_incts() and fmp_balts
#MOD 02/08/26 8:58 AM

def df_to_markdown_with_metadata(df, filename):
    """
    Save DataFrame with metadata to markdown file.
    Transposes DataFrame so dates appear as columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with metadata in df.attrs['metadata']
    filename : str
        Output filepath for markdown file
    """
    md = "# Company Metadata\n\n"
    
    # Add symbol if available
    if 'symbol' in df.attrs:
        md += f"**Symbol:** {df.attrs['symbol']}  \n"
    
    # Add period if available
    if 'period' in df.attrs:
        md += f"**Period:** {df.attrs['period']}  \n"
    
    md += "\n"
    
    # Add company metadata
    if 'metadata' in df.attrs:
        for k, v in df.attrs['metadata'].items():
            md += f"**{k}:** {v}  \n"
    
    md += "\n## Financial Data\n\n"
    md += df.T.to_markdown()
    
    with open(filename, 'w') as f:
        f.write(md)
    
    print(f"Saved to {filename}")
#-----------------------------------------------------	
#MOD 2/8/26 9:41AM
def fmp_inctsFMT(df):
    r"""
    Formats a fmp_incts() DataFrame into an interactive HTML report.

    This function transforms income statement data into a professional HTML document with
    automatic scaling, financial hierarchy styling, and Chart.js for interactive row-level trend analysis.

    Args:
        df (pd.DataFrame): Income statement data from fmp_incts() (date index, metrics as columns).

    Returns:
        None: Writes file to disk and opens it in the default web browser.

    Notes:
        ### Interactive Controls
        - Toggle % of Revenue button shows/hides percentage rows
        - Export to Excel button downloads the visible data as Excel file
        
        ### Scaling Logic
        The function determines the scale based on the first period's 'revenue':
        - If Revenue ≥ 1,000,000,000, values are divided by 1,000,000 (Millions).
        - Otherwise, values are divided by 1,000 (Thousands).

        ### Formulas & Metrics
        % of Revenue rows are calculated as:
        - Line Item % = (Line Item Value / Revenue) × 100
        
        ### Company Metadata
        Displays symbol, period, and company name from df.attrs if available.
    """
    friendly_names = {
        'reportedCurrency': 'Currency', 
        'calendarYear': 'Year', 
        'period': 'Quarter',
        'duplicate_period': 'Duplicate Period',
        'revenue': 'Revenue',
        'costOfRevenue': 'Cost of Revenue',
        'grossProfit': 'Gross Profit',
        'researchAndDevelopmentExpenses': 'R&D Expenses',
        'generalAndAdministrativeExpenses': 'G&A Expenses',
        'sellingAndMarketingExpenses': 'Sales & Marketing Expenses',
        'sga': 'SG&A',
        'otherExpenses': 'Other Expenses',
        'operatingExpenses': 'Operating Expenses',
        'depreciationAndAmortization': 'Depreciation & Amortization',
        'operatingIncome': 'Operating Income',
        'ebitda': 'EBITDA',
        'totalOtherIncomeExpensesNet': 'Total Other Income/Expenses',
        'interestIncome': 'Interest Income',
        'interestExpense': 'Interest Expense',
        'incomeBeforeTax': 'Income Before Tax',
        'incomeTaxExpense': 'Income Tax Expense',
        'netIncome': 'Net Income',
        'eps': 'EPS (Basic)',
        'epsdiluted': 'EPS (Diluted)',
        'weightedAverageShsOut': 'Weighted Avg Shares Outstanding',
        'weightedAverageShsOutDil': 'Weighted Avg Shares Outstanding (Diluted)',
        'fillingDate': 'Filing Date'
    }

    # Extract metadata for header
    symbol = df.attrs.get('symbol', 'N/A')
    period_type = df.attrs.get('period', 'N/A')
    metadata = df.attrs.get('metadata', {})
    company_name = metadata.get('companyName', symbol)
    
    df_work = df.copy()
    
    # Drop validation delta columns
    delta_cols = [col for col in df_work.columns if '_delta_rpt_vs_calc' in col]
    df_work = df_work.drop(columns=delta_cols, errors='ignore')
    
    # Drop fillingDate if present
    df_work = df_work.drop(columns=['fillingDate'], errors='ignore')
    
    # Transpose so dates are columns
    df_work = df_work.T
    
    # Determine Scale based on Revenue
    revenue_val = pd.to_numeric(df_work.loc['revenue', df_work.columns[0]], errors='coerce')
    scale_factor = 1_000_000 if revenue_val >= 1_000_000_000 else 1_000
    scale_label = 'Millions' if scale_factor == 1_000_000 else 'Thousands'

    # ALWAYS ADD % OF REVENUE (will be toggled via button)
    pct_tags = [
        'costOfRevenue', 'grossProfit', 'researchAndDevelopmentExpenses',
        'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses',
        'sga', 'otherExpenses', 'operatingExpenses', 'depreciationAndAmortization',
        'operatingIncome', 'ebitda', 'totalOtherIncomeExpensesNet',
        'interestIncome', 'interestExpense', 'incomeBeforeTax',
        'incomeTaxExpense', 'netIncome'
    ]
    
    rows_to_insert = []
    for tag in pct_tags:
        if tag in df_work.index:
            pct_row = pd.Series(index=df_work.columns, name=f"{tag}_pct", dtype=float)
            for col in df_work.columns:
                val = pd.to_numeric(df_work.loc[tag, col], errors='coerce')
                revenue = pd.to_numeric(df_work.loc['revenue', col], errors='coerce')
                pct_row[col] = (val / revenue * 100) if revenue != 0 else np.nan
            rows_to_insert.append((tag, pct_row))

    # Insert percentage rows in reverse order to maintain positioning
    for parent_tag, pct_row in reversed(rows_to_insert):
        if parent_tag in df_work.index:
            idx_pos = df_work.index.get_loc(parent_tag)
            df_top = df_work.iloc[:idx_pos+1]
            df_bottom = df_work.iloc[idx_pos+1:]
            pct_row.name = f"% of Revenue ({friendly_names.get(parent_tag, parent_tag)})"
            df_work = pd.concat([df_top, pd.DataFrame([pct_row]), df_bottom])

    # Initial Scaling for main numeric rows (exclude metadata and calculated %)
    meta_rows = ['reportedCurrency', 'calendarYear', 'period', 'duplicate_period']
    for row in df_work.index:
        if row not in meta_rows and "%" not in str(row) and row not in ['eps', 'epsdiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil']:
            for col in df_work.columns:
                val = pd.to_numeric(df_work.loc[row, col], errors='coerce')
                if pd.notna(val):
                    df_work.loc[row, col] = val / scale_factor

    # Apply UI Hierarchy
    grand_totals = ['Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EBITDA']
    subtotals = ['Operating Expenses', 'Income Before Tax', 'SG&A']

    def apply_styles(label):
        clean = friendly_names.get(label, label)
        
        if clean in grand_totals: 
            return f"<strong>{clean}</strong>"
        if clean in subtotals: 
            return f"&nbsp;&nbsp;{clean}"
        if "%" in str(label): 
            return f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<em>{label}</em>"
        if clean in ['EPS (Basic)', 'EPS (Diluted)', 'Weighted Avg Shares Outstanding', 'Weighted Avg Shares Outstanding (Diluted)']:
            return f"&nbsp;&nbsp;{clean}"
        return f"&nbsp;&nbsp;&nbsp;&nbsp;{clean}"

    df_work.index = [apply_styles(idx) for idx in df_work.index]

    # Cell Value Formatting
    for idx in df_work.index:
        for col in df_work.columns:
            val = df_work.loc[idx, col]
            if isinstance(val, (int, float)) and pd.notna(val):
                # Different precision for EPS vs other values
                if 'EPS' in str(idx):
                    fmt = "{:.2f}"
                elif "%" in str(idx):
                    fmt = "{:.2f}"
                elif 'Shares' in str(idx):
                    fmt = "{:,.0f}"
                else:
                    fmt = "{:,.2f}"
                df_work.loc[idx, col] = fmt.format(val)

    # Format column headers (dates)
    df_work.columns = [pd.to_datetime(col).strftime('%Y-%m-%d') if pd.notna(pd.to_datetime(col, errors='coerce')) else str(col) for col in df_work.columns]

    # HTML Output with Charting Logic
    table_html = df_work.to_html(classes='income-statement', border=0, escape=False)
    
    # Generate unique filename based on symbol and timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{symbol}_income_statement_{timestamp}.html"
    
    styled_html = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
    <style>
        body {{ font-family: -apple-system, system-ui, sans-serif; padding: 40px; background-color: #f0f2f5; }}
        .container {{ background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .metadata {{ color: #666; font-size: 14px; margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 6px; }}
        .metadata strong {{ color: #1a1a1a; }}
        .controls {{ margin-bottom: 20px; padding: 15px; background-color: #e3f2fd; border-radius: 6px; display: flex; gap: 10px; }}
        .btn {{ padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 600; transition: all 0.2s; }}
        .btn-primary {{ background-color: #2196F3; color: white; }}
        .btn-primary:hover {{ background-color: #1976D2; }}
        .btn-success {{ background-color: #4CAF50; color: white; }}
        .btn-success:hover {{ background-color: #45a049; }}
        h2 {{ color: #1a1a1a; border-bottom: 3px solid #2196F3; padding-bottom: 15px; margin-bottom: 20px; }}
        .income-statement {{ border-collapse: collapse; width: 100%; font-size: 13px; color: #333; }}
        .income-statement thead th {{ background-color: #2196F3; color: white; padding: 12px 10px; text-align: right; text-transform: uppercase; letter-spacing: 1px; font-size: 11px; }}
        .income-statement tbody th {{ text-align: left !important; padding: 8px 15px; border: 1px solid #e0e0e0; border-right: 2px solid #ccc; white-space: nowrap; cursor: pointer; font-family: "SF Mono", monospace; background-color: #fff; }}
        .income-statement tbody th:hover {{ background-color: #e3f2fd; color: #1565c0; }}
        .income-statement tbody td {{ border: 1px solid #e0e0e0; padding: 8px; text-align: right; }}
        .income-statement tbody tr:nth-child(even) {{ background-color: #fafafa; }}
        .income-statement tbody tr.pct-row {{ display: none; }}
        strong {{ font-weight: 800 !important; color: #000; }}
        em {{ color: #777; font-size: 0.85em; }}
        #chartModal {{ display: none; position: fixed; z-index: 100; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); }}
        .modal-content {{ background: white; margin: 5% auto; padding: 20px; border-radius: 12px; width: 80%; max-width: 900px; }}
        .close {{ color: #aaa; float: right; font-size: 28px; font-weight: bold; cursor: pointer; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="metadata">
            <strong>Company:</strong> {company_name} ({symbol}) &nbsp;&nbsp;|&nbsp;&nbsp;
            <strong>Period:</strong> {period_type.upper()}
        </div>
        <div class="controls">
            <button class="btn btn-primary" id="pctToggle" onclick="togglePercentages()">Show % of Revenue</button>
            <button class="btn btn-success" onclick="exportToExcel()">Export to Excel</button>
        </div>
        <h2>Income Statement (in {scale_label})</h2>
        <div style="overflow-x: auto;">{table_html}</div>
    </div>
    <div id="chartModal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <canvas id="rowChart"></canvas>
        </div>
    </div>
    <script>
        let pctVisible = false;
        
        function togglePercentages() {{
            pctVisible = !pctVisible;
            const pctRows = document.querySelectorAll('.income-statement tbody tr');
            const btn = document.getElementById('pctToggle');
            
            pctRows.forEach(row => {{
                const th = row.querySelector('th');
                if (th && th.innerHTML.includes('% of Revenue')) {{
                    row.style.display = pctVisible ? 'table-row' : 'none';
                }}
            }});
            
            btn.textContent = pctVisible ? 'Hide % of Revenue' : 'Show % of Revenue';
        }}
        
        function exportToExcel() {{
            const table = document.querySelector('.income-statement');
            const rows = Array.from(table.querySelectorAll('tr')).filter(row => {{
                // Only include visible rows
                return row.style.display !== 'none';
            }});
            
            const data = rows.map(row => {{
                return Array.from(row.querySelectorAll('th, td')).map(cell => {{
                    // Clean HTML tags and get text
                    const text = cell.innerText || cell.textContent;
                    // Try to convert to number if it looks like one
                    const cleaned = text.replace(/,/g, '');
                    const num = parseFloat(cleaned);
                    return isNaN(num) ? text : num;
                }});
            }});
            
            const ws = XLSX.utils.aoa_to_sheet(data);
            const wb = XLSX.utils.book_new();
            XLSX.utils.book_append_sheet(wb, ws, "Income Statement");
            XLSX.writeFile(wb, "{symbol}_income_statement.xlsx");
        }}
        
        let myChart = null;
        document.querySelectorAll(".income-statement tbody th").forEach(header => {{
            header.onclick = function() {{
                const row = this.parentElement;
                const label = this.innerText.trim();
                const labels = Array.from(document.querySelectorAll(".income-statement thead th")).slice(1).map(th => th.innerText);
                const data = Array.from(row.querySelectorAll("td")).map(td => parseFloat(td.innerText.replace(/,/g, '')));
                document.getElementById("chartModal").style.display = "block";
                if (myChart) myChart.destroy();
                myChart = new Chart(document.getElementById('rowChart'), {{
                    type: 'line', 
                    data: {{ 
                        labels, 
                        datasets: [{{ 
                            label, 
                            data, 
                            borderColor: '#2196F3', 
                            backgroundColor: 'rgba(33,150,243,0.1)', 
                            fill: true, 
                            tension: 0.3 
                        }}] 
                    }},
                    options: {{
                        responsive: true,
                        plugins: {{
                            title: {{
                                display: true,
                                text: label
                            }}
                        }}
                    }}
                }});
            }};
        }});
        function closeModal() {{ document.getElementById("chartModal").style.display = "none"; }}
    </script>
</body>
</html>
"""
    with open(filename, 'w', encoding='utf-8') as f: 
        f.write(styled_html)
    webbrowser.open('file://' + os.path.abspath(filename))
    print(f"Saved to: {filename}")
#-----------------------------------------------------------------
## MOD  01/18/2026 7:00 AM

def fmp_inctsar(sym, facs=None, period='quarter', limit=400):
    '''
    Retrieves raw XBRL "As-Reported" Income Statement data from FMP v3.
    Designed for auditing and reference of raw filing tags.
    
    Parameters:
    -----------
    sym : str
        The stock ticker symbol (e.g., 'AAPL', 'INTC').
    facs : list, optional
        List of specific raw XBRL tags to return.
        NOTE: Tags vary by company/industry. To see available tags for a ticker:
              >>> df = fmp_inctsar('INTC')
              >>> print(df.columns.tolist())
    period : str, default 'quarter'
        The time interval. Options: 'quarter', 'annual' (or 'year').
    limit : int, default 400
        Number of historical filings to retrieve.

    Returns:
    --------
    pd.DataFrame
        DataFrame indexed by date (oldest to newest).
    '''
    sym = sym.upper().strip()
    
    # Normalize period to 'annual' or 'quarter' per API requirements
    p = "annual" if period.lower() in ("year", "annual") else "quarter"

    url = f"https://financialmodelingprep.com/api/v3/income-statement-as-reported/{sym}?period={p}&limit={int(limit)}&apikey={apikey}"
    url = requote_uri(url)

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))

        if not isinstance(stuff, list) or len(stuff) == 0:
            return pd.DataFrame()

        # Load into DataFrame
        df = pd.DataFrame(stuff)

        # 1. Standardize the Index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors="coerce")
            df = df.dropna(subset=['date']).set_index('date')
            df.index.name = "date"
        
        # 2. Chronological Sort
        df = df.sort_index()

        # 3. CLEANUP: Prune columns that are 100% NaN (unused XBRL tags)
        # This is vital for AR data where hundreds of empty columns may exist
        df = df.dropna(axis=1, how='all')

        # 4. PASSIVE VERIFICATION: Flag Operating Income discrepancies
        # Note: We use .fillna(0) only for the math check
        check_tags = ['revenue', 'costofrevenue', 'operatingexpenses', 'operatingincome']
        if all(col in df.columns for col in check_tags):
            df['op_income_error_delta'] = (
                df['revenue'].fillna(0) - 
                df['costofrevenue'].fillna(0) - 
                df['operatingexpenses'].fillna(0) - 
                df['operatingincome'].fillna(0)
            )

        # 5. Factor Filtering
        if facs is not None:
            # Date is excluded from filtering as it is now the index
            clean_facs = [f for f in facs if f != 'date']
            df = df.reindex(columns=clean_facs)

        return df

    except Exception as e:
        print(f"Error in fmp_inctsar for {sym}: {e}")
        return pd.DataFrame()


#--------------------------------------------------------
# 2026-02-08 09:01:12

def fmp_cashfts(sym, facs=None, period='quarter', limit=400, save_md=None):
    '''
    Retrieves standardized Cash Flow Statement data from the FMP v3 endpoint.
    Displays FMP reported totals and validates against calculated component sums.
    
    Parameters:
    -----------
    sym : str
        The stock ticker symbol (e.g., 'AAPL').
    
    period : str, default='quarter'
        The reporting period to retrieve:
        - 'quarter' : Quarterly financial statements (default)
        - 'annual'  : Annual financial statements (FY only)
        - 'ttm'     : Trailing Twelve Months (rolling 4-quarter sum)
    
    limit : int, default=400
        Maximum number of periods to retrieve from FMP.
        - For 'quarter' and 'annual': Returns up to `limit` periods
        - For 'ttm': Returns approximately `limit - 3` TTM periods
          (Formula: ttm_periods = limit - 3, so limit = desired_ttm_periods + 3)
          Examples: limit=8 → 5 TTM periods, limit=23 → 20 TTM periods
    
    facs : list, optional
        List of specific line items to return. If None, returns the full standard set.
        
        Mandatory Columns (Always Returned):
        ['reportedCurrency', 'calendarYear', 'period']
        
        AVAILABLE TAGS:
        Operating Activities Components:
        ['netIncome', 'depreciationAndAmortization', 'deferredIncomeTax', 
         'stockBasedCompensation', 'changeInWorkingCapital', 'accountsReceivables', 
         'inventory', 'accountsPayables', 'otherWorkingCapital', 'otherNonCashItems']
        
        Investing Activities Components:
        ['investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 
         'purchasesOfInvestments', 'salesMaturitiesOfInvestments', 'otherInvestingActivites']
        
        Financing Activities Components:
        ['debtRepayment', 'commonStockIssued', 'commonStockRepurchased', 
         'dividendsPaid', 'otherFinancingActivites']
        
        Cash Reconciliation:
        ['effectOfForexChangesOnCash', 'netChangeInCash', 
         'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod']
        
        FMP Reported Totals (from 10-Q/10-K):
        ['netCashProvidedByOperatingActivities', 'netCashUsedForInvestingActivites',
         'netCashUsedProvidedByFinancingActivities', 'operatingCashFlow', 
         'capitalExpenditure', 'freeCashFlow']
        
        Validation Deltas (Reported - Calculated from Components):
        ['netCashProvidedByOperatingActivities_delta_rpt_vs_calc',
         'netCashUsedForInvestingActivites_delta_rpt_vs_calc',
         'netCashUsedProvidedByFinancingActivities_delta_rpt_vs_calc',
         'netChangeInCash_delta_rpt_vs_calc',
         'freeCashFlow_delta_rpt_vs_calc',
         'cashAtEndOfPeriod_delta_rpt_vs_calc']
        
        Data Quality Flags:
        ['duplicate_period'] - True if multiple filings exist for same calendarYear/period
        
        Metadata:
        ['fillingDate']
    
    save_md : str, optional
        Filepath to save DataFrame with metadata as markdown file.
        If provided, saves output with company metadata header.
        Example: save_md='apple_cash_flow.md'
    
    Returns:
    --------
    pd.DataFrame
        Cash flow statement data with date index and symbol stored in df.attrs['symbol']
        Company metadata stored in df.attrs['metadata'] containing:
        ['companyName', 'cik', 'isin', 'cusip', 'currency', 'exchangeShortName',
         'industry', 'sector', 'country']
        Period type stored in df.attrs['period'] ('quarter', 'annual', or 'ttm')
    
    LLM Context for Validation Columns:
    ------------------------------------
    This dataset includes validation columns (suffix: _delta_rpt_vs_calc) that verify 
    cash flow statement internal consistency.
    
    Formula: Δ = FMP_reported_value - calculated_from_components
    
    Validation Columns & Their Formulas:
    - netCashProvidedByOperatingActivities_delta_rpt_vs_calc = 
        netCashProvidedByOperatingActivities(rpt) - (netIncome + depreciationAndAmortization + 
        deferredIncomeTax + stockBasedCompensation + changeInWorkingCapital + otherNonCashItems)
    - netCashUsedForInvestingActivites_delta_rpt_vs_calc = 
        netCashUsedForInvestingActivites(rpt) - (investmentsInPropertyPlantAndEquipment + 
        acquisitionsNet + purchasesOfInvestments + salesMaturitiesOfInvestments + otherInvestingActivites)
    - netCashUsedProvidedByFinancingActivities_delta_rpt_vs_calc = 
        netCashUsedProvidedByFinancingActivities(rpt) - (debtRepayment + commonStockIssued + 
        commonStockRepurchased + dividendsPaid + otherFinancingActivites)
    - netChangeInCash_delta_rpt_vs_calc = 
        netChangeInCash(rpt) - (netCashProvidedByOperatingActivities + netCashUsedForInvestingActivites + 
        netCashUsedProvidedByFinancingActivities + effectOfForexChangesOnCash)
    - freeCashFlow_delta_rpt_vs_calc = 
        freeCashFlow(rpt) - (operatingCashFlow + capitalExpenditure)
    - cashAtEndOfPeriod_delta_rpt_vs_calc = 
        cashAtEndOfPeriod(rpt) - (cashAtBeginningOfPeriod + netChangeInCash)
    
    Interpreting Validation Deltas:
    Δ = 0        → Components reconcile perfectly with reported total ✓
    Δ > 0        → Reported total exceeds component sum
                   CAUSE: Component breakdown not disclosed in filing
                   EXAMPLE: Company reports "Other Non-Cash Items" as single line without detail
    Δ < 0        → Component sum exceeds reported total
                   CAUSE: Potential data quality issue or parsing error 🚩
    Small Δ      → Rounding differences (typically <0.1% of operating cash flow)
    
    Special Note on cashAtEndOfPeriod_delta_rpt_vs_calc:
    For quarterly and annual periods, this should ALWAYS equal zero in valid financial statements.
    For TTM periods, validation deltas are calculated BEFORE rolling sums, so deltas represent
    the most recent quarter's reconciliation quality.
    Non-zero values indicate data quality issues requiring manual review.
    
    Common Patterns:
    1. netCashProvidedByOperatingActivities_delta_rpt_vs_calc > 0 with otherNonCashItems = 0
       → Company doesn't break out all operating activity adjustments (normal)
    
    2. netChangeInCash_delta_rpt_vs_calc ≠ 0
       → Check if all three activity sections reconcile individually first
    
    3. cashAtEndOfPeriod_delta_rpt_vs_calc ≠ 0
       → Flag for immediate manual review 🚩
    
    When asked to verify data quality:
    1. Check cashAtEndOfPeriod_delta_rpt_vs_calc first (must be 0 for quarterly/annual)
    2. For other deltas, check if delta = 0 (perfect reconciliation)
    3. If delta > 0, check if components exist (look for zeros)
    4. If components are zero, delta indicates missing breakdown (expected)
    5. If components exist but don't sum, flag as data quality issue
    6. Calculate delta as % of reported value to assess materiality
    
    Data Quality Flag - duplicate_period:
    - False: Single filing for this period (normal)
    - True: Multiple filings exist for same calendarYear/period combination
           CAUSES: Amended filings, restatements, or FMP data quality issues
           ACTION: Review fillingDate to identify most recent filing, or 
                   manually verify which filing to use
    
    TTM Period Notes:
    - Cash flow line items are summed over rolling 4 quarters
    - duplicate_period flag not applicable (TTM aggregates across periods)
    - cashAtEndOfPeriod and cashAtBeginningOfPeriod use the most recent quarter values (not summed)
    - Validation deltas calculated on quarterly data BEFORE rolling sums
    - TTM rows show most recent quarter's validation quality
    
    Metadata Notes:
    - Company metadata is fetched via fmp_profF() and attached to df.attrs['metadata']
    - Metadata persists in Python session but not in all file formats
    - Use save_md parameter to export with metadata preserved in markdown format
    - Access metadata: df.attrs['metadata']
    - Period type: df.attrs['period']
    
    Examples:
    ---------
    >>> # Get last 8 quarters
    >>> df = fmp_cashfts('AAPL', period='quarter', limit=8)
    
    >>> # Get last 5 years annual
    >>> df = fmp_cashfts('AAPL', period='annual', limit=5)
    
    >>> # Get last 20 TTM periods
    >>> df = fmp_cashfts('AAPL', period='ttm', limit=23)
    
    >>> # Get specific fields only
    >>> df = fmp_cashfts('AAPL', facs=['operatingCashFlow', 'freeCashFlow', 'capitalExpenditure'])
    
    >>> # Check for duplicate filings
    >>> df = fmp_cashfts('AZTA', period='quarter', limit=8)
    >>> if df['duplicate_period'].any():
    >>>     print("Warning: Multiple filings detected")
    >>>     print(df[df['duplicate_period']][['calendarYear', 'period', 'fillingDate']])
    
    >>> # Verify cash reconciliation
    >>> df = fmp_cashfts('AAPL')
    >>> if df['cashAtEndOfPeriod_delta_rpt_vs_calc'].abs().max() > 0:
    >>>     print("WARNING: Cash reconciliation doesn't balance!")
    
    >>> # Access company metadata
    >>> df = fmp_cashfts('AZTA')
    >>> print(df.attrs['metadata'])
    >>> print(df.attrs['period'])
    
    >>> # Save with metadata to markdown
    >>> df = fmp_cashfts('AZTA', save_md='azenta_cash_flow.md')
    '''
    
    sym = sym.upper().strip()
    
    # Fetch company profile metadata
    try:
        profile = fmp_profF(sym)
        metadata = {
            'companyName': profile.get('companyName'),
            'cik': profile.get('cik'),
            'isin': profile.get('isin'),
            'cusip': profile.get('cusip'),
            'currency': profile.get('currency'),
            'exchangeShortName': profile.get('exchangeShortName'),
            'industry': profile.get('industry'),
            'sector': profile.get('sector'),
            'country': profile.get('country')
        }
    except Exception as e:
        print(f"Warning: Could not fetch metadata for {sym}: {e}")
        metadata = {}
    
    fetch_period = 'quarter' if period.lower() == 'ttm' else period
    
    # URL construction
    url = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{sym}?period={fetch_period}&limit={limit}&apikey={apikey}'
    url = requote_uri(url)

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))
        if not stuff or not isinstance(stuff, list):
            return pd.DataFrame()

        df = pd.DataFrame(stuff)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = 'date'
        
        df = df.sort_index()
        
        # Drop columns that are 100% NaN
        df = df.dropna(axis=1, how='all')
        
        # Flag duplicate periods (for quarter and annual only, not TTM)
        if period.lower() in ['quarter', 'annual']:
            df['duplicate_period'] = df.duplicated(subset=['calendarYear', 'period'], keep=False)
        
        # Calculate internal validation values BEFORE TTM rolling sums
        # Operating Activities
        df['netCashProvidedByOperatingActivities_internal'] = (
            df['netIncome'].fillna(0) +
            df['depreciationAndAmortization'].fillna(0) +
            df['deferredIncomeTax'].fillna(0) +
            df['stockBasedCompensation'].fillna(0) +
            df['changeInWorkingCapital'].fillna(0) +
            df['otherNonCashItems'].fillna(0)
        )
        
        # Investing Activities
        df['netCashUsedForInvestingActivites_internal'] = (
            df['investmentsInPropertyPlantAndEquipment'].fillna(0) +
            df['acquisitionsNet'].fillna(0) +
            df['purchasesOfInvestments'].fillna(0) +
            df['salesMaturitiesOfInvestments'].fillna(0) +
            df['otherInvestingActivites'].fillna(0)
        )
        
        # Financing Activities
        df['netCashUsedProvidedByFinancingActivities_internal'] = (
            df['debtRepayment'].fillna(0) +
            df['commonStockIssued'].fillna(0) +
            df['commonStockRepurchased'].fillna(0) +
            df['dividendsPaid'].fillna(0) +
            df['otherFinancingActivites'].fillna(0)
        )
        
        # Net Change in Cash
        df['netChangeInCash_internal'] = (
            df['netCashProvidedByOperatingActivities'].fillna(0) +
            df['netCashUsedForInvestingActivites'].fillna(0) +
            df['netCashUsedProvidedByFinancingActivities'].fillna(0) +
            df['effectOfForexChangesOnCash'].fillna(0)
        )
        
        # Free Cash Flow
        df['freeCashFlow_internal'] = (
            df['operatingCashFlow'].fillna(0) +
            df['capitalExpenditure'].fillna(0)
        )
        
        # Cash at End of Period
        df['cashAtEndOfPeriod_internal'] = (
            df['cashAtBeginningOfPeriod'].fillna(0) +
            df['netChangeInCash'].fillna(0)
        )
        
        # Calculate validation deltas on quarterly data
        delta_mappings = {
            'netCashProvidedByOperatingActivities': 'netCashProvidedByOperatingActivities_internal',
            'netCashUsedForInvestingActivites': 'netCashUsedForInvestingActivites_internal',
            'netCashUsedProvidedByFinancingActivities': 'netCashUsedProvidedByFinancingActivities_internal',
            'netChangeInCash': 'netChangeInCash_internal',
            'freeCashFlow': 'freeCashFlow_internal',
            'cashAtEndOfPeriod': 'cashAtEndOfPeriod_internal'
        }
        
        for reported_col, calc_col in delta_mappings.items():
            delta_col = f'{reported_col}_delta_rpt_vs_calc'
            df[delta_col] = df[reported_col].fillna(0) - df[calc_col].fillna(0)
        
        # Drop temporary internal calculation columns
        calc_cols = [col for col in df.columns if col.endswith('_internal')]
        df = df.drop(columns=calc_cols)
        
        # TTM period calculations (AFTER validation deltas calculated)
        if period.lower() == 'ttm':
            df = df.reset_index()
            window_size = df['period'].nunique() if df['period'].nunique() in [2, 4] else 4
            
            # Identify columns to sum vs. keep as-is
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Don't sum these point-in-time values
            point_in_time_cols = ['cashAtEndOfPeriod', 'cashAtBeginningOfPeriod']
            
            # Don't sum Year or validation deltas (deltas represent quarterly reconciliation quality)
            cols_to_sum = [c for c in numeric_cols if 'Year' not in c and 
                          c not in point_in_time_cols and
                          '_delta_rpt_vs_calc' not in c]
            
            # Sum flow items
            df[cols_to_sum] = df[cols_to_sum].rolling(window=window_size).sum()
            
            # For cash balances, keep most recent value (point-in-time, not cumulative)
            # cashAtEndOfPeriod already has most recent value
            # cashAtBeginningOfPeriod needs to reference 4 quarters ago
            if 'cashAtBeginningOfPeriod' in df.columns:
                df['cashAtBeginningOfPeriod'] = df['cashAtBeginningOfPeriod'].shift(window_size - 1)
            
            # Validation deltas stay as-is (represent most recent quarter's reconciliation)
            
            # Drop incomplete rolling windows
            df = df.dropna(subset=[cols_to_sum[0]]).set_index('date')
        
        elif period.lower() == 'annual':
            df = df[df['period'] == 'FY']
        
        # Define standard column order
        standard_order = [
            'reportedCurrency', 'calendarYear', 'period', 'duplicate_period',
            # Operating Activities
            'netIncome', 'depreciationAndAmortization', 'deferredIncomeTax',
            'stockBasedCompensation', 'changeInWorkingCapital', 
            'accountsReceivables', 'inventory', 'accountsPayables', 
            'otherWorkingCapital', 'otherNonCashItems',
            'netCashProvidedByOperatingActivities', 'operatingCashFlow',
            # Investing Activities
            'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet',
            'purchasesOfInvestments', 'salesMaturitiesOfInvestments',
            'otherInvestingActivites', 'netCashUsedForInvestingActivites',
            # Financing Activities
            'debtRepayment', 'commonStockIssued', 'commonStockRepurchased',
            'dividendsPaid', 'otherFinancingActivites',
            'netCashUsedProvidedByFinancingActivities',
            # Cash Reconciliation
            'effectOfForexChangesOnCash', 'netChangeInCash',
            'cashAtBeginningOfPeriod', 'cashAtEndOfPeriod',
            # Derived Metrics
            'capitalExpenditure', 'freeCashFlow',
            # Validation Deltas
            'netCashProvidedByOperatingActivities_delta_rpt_vs_calc',
            'netCashUsedForInvestingActivites_delta_rpt_vs_calc',
            'netCashUsedProvidedByFinancingActivities_delta_rpt_vs_calc',
            'netChangeInCash_delta_rpt_vs_calc',
            'freeCashFlow_delta_rpt_vs_calc',
            'cashAtEndOfPeriod_delta_rpt_vs_calc',
            # Metadata
            'fillingDate'
        ]
        
        mandatory_cols = ['reportedCurrency', 'calendarYear', 'period']
        
        # Build final column list
        if facs is None:
            final_cols = mandatory_cols.copy()
            for col in standard_order:
                if col in df.columns and col not in final_cols:
                    final_cols.append(col)
        else:
            clean_facs = [f for f in facs if f in df.columns and f not in (mandatory_cols + ['date', 'symbol'])]
            final_cols = mandatory_cols.copy()
            for col in standard_order:
                if col in clean_facs and col not in final_cols:
                    final_cols.append(col)
            for col in clean_facs:
                if col not in final_cols:
                    final_cols.append(col)
        
        result_df = df[final_cols]
        result_df.attrs['symbol'] = sym
        result_df.attrs['metadata'] = metadata
        result_df.attrs['period'] = period.lower()
        
        # Save to markdown if requested
        if save_md:
            df_to_markdown_with_metadata(result_df, save_md)
        
        return result_df

    except Exception as e:
        print(f"Error in fmp_cashfts for {sym}: {e}")
        return pd.DataFrame()
#-----------------------------------------------------
# 2026-01-18 07:55:35
def fmp_cashftsar(sym, facs=None, period='quarter', limit=400):
    '''
    Retrieves raw XBRL "As-Reported" Cash Flow Statement data from FMP v3.
    '''
    sym = sym.upper().strip()
    p = "annual" if period.lower() in ("year", "annual") else "quarter"
    # URL updated with limit parameter
    url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement-as-reported/{sym}?period={p}&limit={int(limit)}&apikey={apikey}"
    url = requote_uri(url)

    try:
        response = urlopen(url, context=ssl_context)
        stuff = json.loads(response.read().decode("utf-8"))
        if not isinstance(stuff, list) or len(stuff) == 0: return pd.DataFrame()

        df = pd.DataFrame(stuff)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.index.name = "date"
        
        df = df.sort_index()
        df = df.dropna(axis=1, how='all')

        # Passive Verification Identity
        check_tags = ['netcashprovidedbyoperatingactivities', 'netcashusedforinvestingactivites', 
                      'netcashusedprovidedbyfinancingactivities', 'netchangeincash']
        if all(col in df.columns for col in check_tags):
            df['cash_flow_error_delta'] = (
                df['netcashprovidedbyoperatingactivities'].fillna(0) + 
                df['netcashusedforinvestingactivites'].fillna(0) + 
                df['netcashusedprovidedbyfinancingactivities'].fillna(0) + 
                df.get('effectofforexchangesoncash', pd.Series(0, index=df.index)).fillna(0) - 
                df['netchangeincash'].fillna(0)
            )

        if facs is not None:
            clean_facs = [f for f in facs if f != 'date']
            df = df.reindex(columns=clean_facs)
        return df

    except Exception as e:
        print(f"Error in fmp_cashftsar for {sym}: {e}")
        return pd.DataFrame()

#----------------------------------------------------------
#MOD 1/20/267:59AM

def fmp_shares(sym, facs=['outstandingShares']):
    '''
    Retrieves historical shares data.
    FIXED: Uses pd.to_numeric to prevent strings from breaking math operations.
    '''
    sym = sym.upper()
    url = f'https://financialmodelingprep.com/api/v4/historical/shares_float?symbol={sym}&apikey={apikey}'

    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)

    try:
        if not stuff:
             return pd.DataFrame(columns=facs)

        idx = pd.to_datetime([sub['date'] for sub in stuff])
        df = pd.DataFrame([[sub.get(k) for k in facs] for sub in stuff], columns=facs, index=idx)
        
        # --- THE FIX: Convert strings to numbers ---
        df = df.apply(pd.to_numeric, errors='coerce')
        
        return df.iloc[::-1]
            
    except (IndexError, KeyError, TypeError):
        return pd.DataFrame(columns=facs)
#------------------------------------------------------
###01/20/26 8:01 AM

def fmp_entts(sym, freq='Q', limit=40):
    """
    Returns historical Enterprise Value DataFrame.
    Includes explicit numeric enforcement to prevent "can't multiply sequence" errors.
    """
    sym = sym.upper()
    
    # 1. Fetch Data
    bs = fmp_balts(sym, period='quarter', limit=limit, 
                   facs=['totalDebt', 'minorityInterest', 'cashAndShortTermInvestments'])
    inct = fmp_incts(sym, period='quarter', limit=limit, 
                    facs=['weightedAverageShsOutDil'])
    sh = fmp_shares(sym, facs=['outstandingShares'])
    px = fmp_price(sym, start=bs.index.min().strftime('%Y-%m-%d'))
    
    # 2. Alignment Logic
    if freq.upper() == 'Q':
        df = pd.concat([bs, inct], axis=1)
        df['price'] = px['close'].reindex(df.index, method='ffill')
        df['shares_actual'] = sh['outstandingShares'].reindex(df.index, method='ffill')
    else:
        df = px[['close']].rename(columns={'close': 'price'})
        df = df.join(bs).join(inct).join(sh['outstandingShares']).ffill()
        df.rename(columns={'outstandingShares': 'shares_actual'}, inplace=True)

    # --- THE DEFENSIVE FIX: FORCE NUMERIC ---
    # This ensures that even if fmp_shares returns a string, it becomes a number here
    cols_to_fix = ['price', 'shares_actual', 'weightedAverageShsOutDil', 
                   'totalDebt', 'minorityInterest', 'cashAndShortTermInvestments']
    
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. EV Calculations (Now safe from TypeErrors)
    df['ev_institutional'] = (df['price'] * df['weightedAverageShsOutDil']) + \
                             (df['totalDebt'].fillna(0) + df['minorityInterest'].fillna(0) - \
                              df['cashAndShortTermInvestments'].fillna(0))
    
    df['ev_actual'] = (df['price'] * df['shares_actual']) + \
                       (df['totalDebt'].fillna(0) + df['minorityInterest'].fillna(0) - \
                        df['cashAndShortTermInvestments'].fillna(0))
    
    return df
#-------------------------------------------------------
#https://financialmodelingprep.com/api/v3/historical-chart/1min/BTCUSD?apikey=

def fmp_intra(sym, period='1hour'):
    """
    sym = single symbol as a string (not case sensitive)
    period = '1day' returns:  'adjClose', 'change', 'changeOverTime', 'changePercent', 'close','high', 'label', 'low', 'open', 'unadjustedVolume', 'volume', 'vwap'
    period = '1min', '5min', '15min', '30min', '1hour' returns:  'close', 'high', 'low', 'open', 'volume'

    """

    if period=='1day':
        url = "https://financialmodelingprep.com/api/v3/historical-price-full/"+sym.upper()+"?apikey="+apikey
        response = urlopen(url, context=ssl_context)
        data = response.read().decode("utf-8")
        return pd.DataFrame(json.loads(data)['historical']).set_index('date').sort_index(ascending=True)
    else:
        url = "https://financialmodelingprep.com/api/v3/historical-chart/"+period+"/"+sym.upper()+"?apikey="+apikey
        return pd.read_json(url).set_index('date').sort_index(ascending=True)
		
#-----------------------------------------------------

def fmp_plot_ts(ts, step=5, figsize=(10,7), title=''):
    """
    plot timeseries ignoring date gaps

    Params
    ------
    ts : pd.DataFrame or pd.Series
    step : int, display interval for ticks
    figsize : tuple, figure size
    title: str
    """

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(ts.dropna().shape[0]), ts.dropna())
    ax.set_title(title)
    ax.set_xticks(np.arange(len(ts.dropna())))
    ax.set_xticklabels(ts.dropna().index.tolist());

    # tick visibility, can be slow for 200,000+ ticks 
    xticklabels = ax.get_xticklabels() # generate list once to speed up function
    for i, label in enumerate(xticklabels):
        if not i%step==0:
            label.set_visible(False)  
    fig.autofmt_xdate()   
    ax.xaxis.grid(True, 'major')		
	
#-----------------------------------------------------

def fmp_search(searchterm):
    '''
    for exchange searches:  ETF | MUTUAL_FUND | COMMODITY | INDEX | CRYPTO | FOREX | TSX | AMEX | NASDAQ | NYSE | EURONEXT
    
    '''
    searchurl='https://financialmodelingprep.com/api/v3/search?query='+searchterm+'&limit=1000&apikey='+apikey
    response = urlopen(searchurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    df= pd.DataFrame(stuff)[['symbol', 'name']]
    pd.set_option('display.max_rows', len(df)+1)
    df.sort_values('name',inplace=True)
    return df.set_index('symbol')	

        
#-------------------------------------------------------


def fmp_mergarb(syms, shar_fact=1, cash=0, start='1960-01-01', per=True):
    '''
    syms:  list of 2 symbols as strings.  acquirer first, acquired second.
           ex: ['CNI', 'KSU']
    shar_fact: float, number of shareds of acquirer that acquired is receiving
    cash: amount of cash per share acquired is recieving
    per: bool.  True returns spread as percentage of acquired,
                False returns spread as a float
    
    returns a df of acquirer price, acquired price, and Arb calculation with 
    columns sym1, sym2, and Arb
      
    
    '''
    df=fmp_multprice(syms, start, facs=['close'])
    if per==True:
        df['arb']= ((df.iloc[:,0]*shar_fact+cash)-df.iloc[:,1])/df.iloc[:,1]
    else:
        df['arb']= df.iloc[:,0]*shar_fact+cash-df.iloc[:,1]   
    return df	
	
#---------------------------------------------------------------------
def fmp_screen(limit=10000, **kwargs):
    """
    Uses the Financial Modeling Prep Screen API to filter companies based on criteria.

    Parameters:
        sector (str, optional): Sector to filter by.
        industry (str, optional): Industry to filter by.
        country (str, optional): Country to filter by.
            
    ex:  fmp_screen(country='US', marketCapMoreThan='1000000000 ', industry='Insurance—Life')
        
    Sectors and Industries
    
    Basic Materials:

Agricultural Inputs
Aluminum
Chemicals
Chemicals - Specialty
Construction Materials
Copper
Gold
Industrial Materials
Other Precious Metals
Paper, Lumber & Forest Products
Silver
Steel

Communication Services:

Advertising Agencies
Broadcasting
Entertainment
Internet Content & Information
Publishing
Telecommunications Services

Consumer Cyclical:

Apparel - Footwear & Accessories
Apparel - Manufacturers
Apparel - Retail
Auto - Dealerships
Auto - Manufacturers
Auto - Parts
Auto - Recreational Vehicles
Department Stores
Furnishings, Fixtures & Appliances
Gambling, Resorts & Casinos
Home Improvement
Leisure
Luxury Goods
Packaging & Containers
Personal Products & Services
Residential Construction
Restaurants
Specialty Retail
Travel Lodging
Travel Services

Consumer Defensive:

Agricultural Farm Products
Beverages - Alcoholic
Beverages - Non-Alcoholic
Beverages - Wineries & Distilleries
Discount Stores
Education & Training Services
Food Confectioners
Food Distribution
Grocery Stores
Household & Personal Products
Packaged Foods
Tobacco

Energy:

Coal
Energy
Oil & Gas Drilling
Oil & Gas Equipment & Services
Oil & Gas Exploration & Production
Oil & Gas Integrated
Oil & Gas Midstream
Oil & Gas Refining & Marketing
Solar
Uranium

Financial Services:

Asset Management
Asset Management - Bonds
Asset Management - Cryptocurrency
Asset Management - Global
Asset Management - Income
Asset Management - Leveraged
Banks
Banks - Diversified
Banks - Regional
Financial - Capital Markets
Financial - Conglomerates
Financial - Credit Services
Financial - Data & Stock Exchanges
Financial - Mortgages
Insurance - Brokers
Insurance - Diversified
Insurance - Life
Insurance - Property & Casualty
Insurance - Reinsurance
Insurance - Specialty
Investment - Banking & Investment Services
Shell Companies

Healthcare:

Biotechnology
Drug Manufacturers - General
Drug Manufacturers - Specialty & Generic
Healthcare
Medical - Care Facilities
Medical - Devices
Medical - Diagnostics & Research
Medical - Distribution
Medical - Equipment & Services
Medical - Healthcare Information Services
Medical - Healthcare Plans
Medical - Instruments & Supplies
Medical - Pharmaceuticals
Medical - Specialties

Industrials:

Aerospace & Defense
Agricultural - Machinery
Air Freight/Couriers
Airlines, Airports & Air Services
Business Equipment & Supplies
Conglomerates
Construction
Consulting Services
Electrical Equipment & Parts
Engineering & Construction
Industrial - Distribution
Industrial - Infrastructure Operations
Industrial - Machinery
Industrial - Pollution & Treatment Controls
Integrated Freight & Logistics
Manufacturing - Metal Fabrication
Manufacturing - Miscellaneous
Manufacturing - Tools & Accessories
Marine Shipping
Railroads
Rental & Leasing Services
Security & Protection Services
Specialty Business Services
Staffing & Employment Services
Trucking
Waste Management
Wholesale Distributors

Real Estate:

REIT - Diversified
REIT - Healthcare Facilities
REIT - Hotel & Motel
REIT - Industrial
REIT - Mortgage
REIT - Office
REIT - Residential
REIT - Retail
REIT - Specialty
Real Estate - Development
Real Estate - Diversified
Real Estate - General
Real Estate - Services

Technology:

Communication Equipment
Computer Hardware
Consumer Electronics
Electronic Gaming & Multimedia
Hardware, Equipment & Parts
Information Technology Services
Internet Software/Services
Semiconductors
Software - Application
Software - Infrastructure
Software - Services
Technology Distributors

Utilities:

Diversified Utilities
Independent Power Producers
Regulated Electric
Regulated Gas
Regulated Water
Renewable Utilities
    
    	Country:  'US', 'CN', 'TW', 'FR', 'CH', 'NL', 'CA', 'JP', 'DK', 'IE', 'AU',
           'GB', 'DE', 'SG', 'BE', 'IN', 'BR', 'ZA', 'AR', 'ES', 'NO', 'HK',
           'IT', 'MX', 'BM', 'LU', 'SE', 'FI', 'CO', 'KR', 'ID', 'JE', 'IL',
           'PT', 'UY', 'CL', 'MC', 'CY', 'MA', 'KY', 'RU', 'PR', 'PH', 'IS',
           'TR', 'IM', 'TH', 'PA', 'PE', 'GG', 'Peru', 'AE', 'NZ', 'GR', 'CR',
           'MY', 'BB', 'BS', 'GA', 'JO', 'VG', 'DO', 'ZM', 'MT', 'CK', 'MN',
           'LT', 'MO', 'AI'
           
    
        Returns:
            pd.DataFrame: A DataFrame with the filtered companies.
        """

    base_url = f"https://financialmodelingprep.com/api/v3/stock-screener"
    
    cols = ['symbol', 'companyName', 'sector', 'industry', 'country', 'marketCap(mil)', 'exchange']

    # Build query parameters dynamically 
    params = {
        k: (v.replace('—', '-') if isinstance(v, str) else v) 
        for k, v in kwargs.items() if v is not None
    }
    
    # --- EXPLICIT ASSIGNMENT: Ensure limit is in params even if not passed in call ---
    params["limit"] = limit
    params["apikey"] = apikey
    
    try:
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Failed to retrieve data. Status code: {response.status_code}")
            return pd.DataFrame(columns=cols).set_index('symbol')
        
        data = response.json()
        
        if not data:
            print(f"No results found for provided filters: {kwargs}")
            return pd.DataFrame(columns=cols).set_index('symbol')
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(columns=cols).set_index('symbol')
    
    print(f"Found {len(data)} stocks:")
    
    df = pd.DataFrame([ 
        [sub.get('symbol'), sub.get('companyName'), sub.get('sector'), 
         sub.get('industry'), sub.get('country'), sub.get('marketCap'), 
         sub.get('exchangeShortName')] 
        for sub in data
    ], columns=['symbol', 'companyName', 'sector', 'industry', 'country', 'marketCap', 'exchange'])

    df['marketCap(mil)'] = (df['marketCap'] / 1000000).round(0)
    df.sort_values('marketCap(mil)', ascending=False, inplace=True)
    df['marketCap(mil)'] = df['marketCap(mil)'].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")

    df = df.drop('marketCap', axis=1)
    df.set_index('symbol', inplace=True)

    return df


#------------------------------------------------------------
def fmp_earnSym(sym, n=5):
    """
    input: symbol as string
           n as int.  the number of quarters to return
    returns:  historical and future earnings dates, times and estimates
    """
    url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{sym}?apikey={apikey}"
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
    return pd.DataFrame(stuff).head(n).sort_values('date')
#------------------------------------------------------------------

def fmp_earnDateNext(sym):
    '''input (str) sym
       returns next earnings date as a string or 'NA' if there's a KeyError or no valid date'''
    
    try:
        df = fmp_earnSym(sym)
        # Convert 'date' column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter rows where 'eps' is NaN
        nan_eps_df = df[df['eps'].isna()]
        
        # Get the earliest 'date' where 'eps' is NaN
        earliest_date = nan_eps_df['date'].min()
        
        # Check if earliest_date is NaT (Not a Time)
        if pd.isna(earliest_date):
            return 'NA'
        
        return earliest_date.strftime('%Y-%m-%d')
    
    except KeyError:
        return 'NA'

#----------------------------------------------------------------------------------
def fmp_earnCal(start=str(dt.datetime.now().date()), end=str(dt.datetime.now().date())):
    '''
    input: start and end dates like 'XXXX-xx-xx'
    returns: a dataframe of symbols with date, time and estimates where available
    
    '''
    earnurl='https://financialmodelingprep.com/api/v3/earning_calendar?from='+start+'&to='+end+'&apikey='+apikey
    response = urlopen(earnurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    df=pd.DataFrame(stuff).iloc[:,0:5]
    df.set_index('symbol', inplace=True)

####  having trouble concatting prof df and df  #######    
    
#    df.date=pd.to_datetime(df.date)
#    name = fmp_prof(df.index.tolist(), facs=['companyName', 'industry', 'mktCap'])
#    fin =  pd.concat([name,df], axis=1)
#    fin.sort_values(['date', 'time'], inplace=True, ascending=[True, False])
    return df
#----------------------------------------------------------------------------------

def fmp_earnEst(sym, period='quarter'):
    '''
    inputs: symbol as str
            period as string.  annual, quarter
            
    returns a dataframe with the following columns: 
    
            'RevLow', 'RevHigh', 'RevAvg', 
            'EbitdaLow', 'EbitdaHigh', 'EbitdaAvg', 'EbitLow', 
            'EbitHigh', 'EbitAvg', 'IncLow', 'IncHigh', 
            'IncAvg', 'SgaExpenseLow', 'SgaExpenseHigh', 
            'SgaExpenseAvg', 'EpsAvg', 'EpsHigh', 'EpsLow', 
            'NumAnalystRev', 'NumAnalystsEps'
    
    
    '''

    url= f"https://financialmodelingprep.com/api/v3/analyst-estimates/{sym}?period={period}&apikey={apikey}"
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data) 

    idx = [d['date'] for d in stuff]


    data = [{k: v for i, (k, v) in enumerate(item.items()) if 2 <= i < 22} for item in stuff]

    df=pd.DataFrame(data=data, index=idx)


    df.columns=columns=['RevLow', 'RevHigh', 'RevAvg', 
                       'EbitdaLow', 'EbitdaHigh', 'EbitdaAvg', 'EbitLow', 
                       'EbitHigh', 'EbitAvg', 'IncLow', 'IncHigh', 
                       'IncAvg', 'SgaExpenseLow', 'SgaExpenseHigh', 
                       'SgaExpenseAvg', 'EpsAvg', 'EpsHigh', 'EpsLow', 
                       'NumAnalystRev', 'NumAnalystsEps']
    return df
#--------------------------------------------------------------------------------------------

def fmp_ticker(sym):
    url = requote_uri('https://financialmodelingprep.com/api/v3/search-ticker?query='+sym+'&limit=1000&apikey='+apikey)
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    return stuff	
	
#---------------------------------------------------------------------------------------

def fmp_scaler(df, names=None):
    df_scaled=pd.DataFrame(StandardScaler().fit_transform(df), 
                columns=names, 
                index=df.index)
    return df_scaled         

#--------------------------------------------------------
def fmp_cumret(df):
    df=df.pct_change()
    df = (1 + df).cumprod() - 1
    df.iloc[0,:]=0
    return df
	
#-------------------------------------------------------------------------------------------

def fmp_cap(sym):
    '''
    Gross Cap is 4 quarters rolling sum of Revenue divided by current quarter Enterprise Value
    params sym: str
    returns: Series with date index and Gross Cap
    '''

    df=fmp_fund(sym, 'income-statement')['revenue']
    dff = fmp_fund(sym, 'enterprise-values')['enterpriseValue']
    condf=pd.concat([df,dff], axis=1)
    condf['cap'] = condf.revenue.rolling(4).sum()/condf.enterpriseValue
    return condf.cap	
#--------------------------------------------------------------------------------------------
def fmp_efficiency(sym):
    '''
    Gross Cap is 4 quarters rolling sum of G&A Exp  divided by current quarter Revenue
    params sym: str
    returns: Series with date index and Gross Cap
    '''

    df=fmp_fund(sym, 'income-statement')['revenue']
    dff = fmp_fund(sym, 'income-statement')["operatingIncome"]
    condf=pd.concat([df,dff], axis=1)
    condf['eff'] = condf.operatingIncome / condf.revenue
    return condf.eff
	
	
#-----------------------------------------------------------------------------------------

def fmp_div(sym, num=5):  
    '''
Declaration Date: This is the date on which the company's board of directors announces 
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
    '''

    url='https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/'+sym+'?apikey='+apikey

    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data) 
    df = pd.DataFrame(stuff['historical'])
    df = df.drop(columns=['label'])
    df = df.rename(columns={'date': 'exDate'})
    df.set_index('exDate', inplace=True)
    df.index = pd.to_datetime(df.index)
    df=df.sort_index()



    df['trail'] = df.adjDividend.rolling(window='359D').sum()
    price=fmp_price(sym, start=df.index[0].strftime('%Y-%m-%d'))
    price.index = pd.to_datetime(price.index)
    newdf = pd.concat([df,price.reindex(df.index)], axis=1)
    newdf['trailYield']= np.round(newdf.trail/newdf.close*100,2)

    num = df['dividend'].rolling(window='360D').count()[-1]

    newdf['curYield']= np.round(newdf.adjDividend*num/newdf.close*100,2)

    
    return newdf

#---------------------------------------------------------------------------------------	
def fmp_idx(syms, weights = None, rebal = 'once', fac= 'adjClose',start='1980-01-01',name='idx'):
    '''
	   syms: list of symbols
       weights: list of weights must = len(syms) and equal 1
       rebal:  'once', 'quarterly', or 'yearly' for rebalance period
       name: a label for the index, string
       returns: res.  The res object in the bt library is typically a bt.run.Result object, and it provides a variety of methods and attributes to 
       analyze the backtest results. Here is a list of some commonly used methods and attributes:

Methods
res.display()

Displays a summary of the performance statistics (e.g., total return, Sharpe ratio, max drawdown, etc.).
res.plot()

Plots the equity curve of the strategy and possibly other metrics depending on options passed.
res.prices

Returns the price series used in the backtest for each asset in the portfolio.
res.weights

Returns a DataFrame of the weights assigned to each asset over time.
res.stats

Returns a list of the performance statistics and other information related to the backtest.
res.get_transactions()

Returns a DataFrame with all the transactions executed during the backtest.
res.get_security_weights()

Returns a dictionary with weights for each security in the portfolio over time.
res.prices.plot()

Plots the price series (if you want to inspect the prices used in the backtest).
res.get_security_returns()

Returns the individual security returns.
Attributes
res.strategy

Provides access to the strategy used in the backtest, which allows you to inspect or modify it.
res.stats

The performance metrics of the strategy, often displayed in a summary format via display().
res.perf

The time series of the portfolio's performance over the backtest period.
res.assets

A list of assets that were included in the backtest.
res.rets

The portfolio’s returns as a time series (returns from one time period to the next).
res.prices

The prices of the assets in the portfolio over time.
res.security_weights

A DataFrame showing the portfolio’s weights in individual assets over time.
res.benchmark

If a benchmark was specified, this will contain benchmark performance data.
res.prices.index

The index (dates) corresponding to the price and portfolio values.
	            
    
    '''
    if weights == None:
        weights=[np.round(1 / len(syms),3)] * len(syms)
        
    rebal_mapping = {
        'once': bt.algos.RunOnce(),
        'quarterly': bt.algos.RunQuarterly(),
        'yearly': bt.algos.RunYearly(),
    }

    if rebal not in rebal_mapping:
        raise ValueError("Invalid value for rebal parameter. Use 'once', 'quarterly', or 'yearly'.")

    rebal_per = rebal_mapping[rebal]   
    
    px=fmp_priceLoop(syms, start=start, fac=fac).dropna()
    
    print ('First available data is '+str(px.index[0].date()))
    print('weights: '+str(dict(zip(syms, weights))))
    print('type:  '+fac)
    idx = bt.Strategy(name, [rebal_per,
                       bt.algos.SelectAll(),
                       bt.algos.WeighSpecified(**dict(zip(syms, weights))),
                       bt.algos.Rebalance()])
    print('Creating Index')
    t = bt.Backtest(idx, px)
    res = bt.run(t)
    print(res.prices.tail(1))
   
    return res
    
#--------------------------------------------------------------------------------------------
def fmp_13F(cik='0001067983', leng=40, date='2022-12-31'):
    
 
    
    '''
   
Inputs: cik# as a string
	    leng as a string = number of symbols to return
        date as string yyyy-mm-dd is one of 4 quarted end dates:  3/31, 6/30, 9/30, or 12/31.  will ususlly not be 
        availabe until AFTER 45 days from quarter end
Output: top 40 holdings df of date of report, symbol, position size in shares 
        and dollars, % of position, and calculated price
		and sorted by position percentage
        
    '''
    
    
    insurl='https://financialmodelingprep.com/api/v3/form-thirteen/'+cik+'?date='+date+'&apikey='+apikey
    response = urlopen(insurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    stuff
    df=pd.DataFrame(stuff, index=[i['tickercusip'] for i in stuff])
    df['bps'] = np.round(df.value/df.value.sum(axis=0)*10000,0)
    df['bps']=df['bps'].astype(int)
    df['px'] = np.round(df.value/df.shares,2)
    df.sort_values(by='bps', ascending=False, inplace=True)
    return df[['date',	'acceptedDate', 'cusip', 
        'nameOfIssuer',	'titleOfClass','shares', 'px', 'value',	 'bps']].head(leng)
		
#--------------------------------------------------------------------------------------------
def fmp_cikToEntity(cik):
    '''
Input:  cik number as a string
Output: Entity name as a string
    
    '''

    url = 'https://financialmodelingprep.com/api/v3/cik/'+cik+'?apikey='+apikey

    # Send request to Financial Modeling Prep API
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    stuff=stuff[0]
    return stuff['name']
#--------------------------------------------------------------------------------------
def fmp_13Fentity(entity='berkshire'):
    
    '''
   
Inputs: entity name as string
	    
Output: dataframe of enbtity name matches and cik # 
    
    '''
       
    insurl='https://financialmodelingprep.com/api/v3/cik-search/'+entity+'?apikey='+apikey
    response = urlopen(insurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    df=pd.DataFrame(stuff)
    return df.loc[:, ['name','cik']].sort_values(by='name')
    


#--------------------------------------------------------------------------------------
def fmp_const(sym, tickersOnly=False):
    '''
    Input:  sym:str --input an etf or mutual fund
    tickersOnly: bool, returns a list of tickers.  default is False
    Output: returns a df of the following columns:
            'name', 'pct', 'price', 'country', 'updated'
    '''
    sym=sym.upper()
    insurl= f'https://financialmodelingprep.com/api/v3/etf-holder/{sym}?apikey={apikey}'
    response = urlopen(insurl)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    df=pd.DataFrame(stuff)
    
    ##extracts 2 letter country code from the front of isin #
    df['country'] = df['isin'].apply(lambda x: re.split('[^a-zA-Z]', x)[0])
    if tickersOnly:
         return df.asset.tolist()
    df = df.loc[:,['asset', 'isin', 'name', 'weightPercentage', 'country','updated']].set_index('asset', drop=True)
    df.columns=['isin','name', 'pct', 'country', 'updated']
    return df		
	
#-----------------------------------------------------------------------------------------------

# def fmp_shares(sym='aapl'):
#     '''
#     input:symbol
#     output: list : float, oustanding, percent float
#     '''
#     sym=sym.upper()
#     floaturl=r'https://financialmodelingprep.com/api/v4/shares_float?symbol='+sym+'&apikey='+apikey
#     response = urlopen(floaturl, context=ssl_context)
#     data = response.read().decode("utf-8")
#     stuff=json.loads(data)
    
#     return [stuff[0]['floatShares'],stuff[0]['outstandingShares'], np.round(stuff[0]['floatShares']/stuff[0]['outstandingShares'],3)] 
	
#-------------------------------------------------------------------------------------------------



def fmp_close(sym,lbk=1000):  
    ####when mkt trading lbk=2 is prev day####
    
    '''returns a 2xn list of dicts {date: date,  close:close} with default length of 1000
	    leng = last n closing prices
	    '''
    #sym=sym.upper()
    closeurl='https://financialmodelingprep.com/api/v3/historical-price-full/'+sym+'?serietype=line&timeseries='+str(lbk)+'&apikey='+apikey
    response = urlopen(closeurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    l=stuff['historical'] 
    l.reverse()
    return l
#-------------------------------------------------------------------------------------------------
	
# def fmpw_returns(syms, lbk=63):
    # '''returns simple return and anual hv of a symbol(s) for a given lookback in days  0
       # syms: list of a single symbol or multiple symbols
       # lkbk: positive int for market days to lookback.  1 yr = 252, 1 mo = 21.
             # most recent quote is a realtime quote during mkt hours so oneday is lkbk=1
             # after market hours for a 1 day return lkbk=1'''
    
    # b=[]  #list of lookback prices for return calc
    # d=[]  #list of hv per symbol
    # e=[]  # modified list of working symbols for next section
    # print('Running Lbk and HV Loop')
    # for i in notebook.tqdm(syms):
        # try:
            # b.append(fmp_close(i, lbk)[-lbk]['close'])
            # d.append(fmpw_hv(i, lbk))
            # e.append(i)
        # except KeyError:
            # print('Symbol '+i+' is not available')
            # continue
        # except IndexError:
            # print('Not enough history available for symbol '+i )
            # continue
    # d=pd.DataFrame(d, index=e, columns=['hv'])    
    # c=[] #list of current prices for return calculation
    # print('Running Most Recent Close Loop')
    # for i in notebook.tqdm(e):
        # c.append(fmp_rt(i))    
    # df=pd.DataFrame(list(zip(c,b)), index=e)
    # df[str(lbk)+'Ret']=np.round(((df.iloc[:,0]/df.iloc[:,1])-1)*100,2)
    # df=df[str(lbk)+'Ret']
    # df=pd.concat([df, d], axis=1)
    # df['sharp'] = np.round(df.iloc[:,0]/df.iloc[:,1],2)
    # return df

#----------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------

    
#------------------------------------------------------------------------------------
    
def fmp_corr(sym1, sym2, lbk=60):
    '''input sy1, sy2 and lbk 
    outputs corr coefficient'''
    X=[sub['close'] for sub in fmp_close(sym1, lbk)]
    Xr=np.diff(np.log(X))
    Y=[sub['close'] for sub in fmp_close(sym2, lbk)]
    Yr=np.diff(np.log(Y))
    return np.round(np.corrcoef(Xr,Yr)[0,1],3)
    
    

#---------------------------------------------------


    
#-----------------------------------------------------------------------------

def fmp_cormatrix(syms, start=60):
    """
    Calculate the correlation matrix or coefficient of log returns for a set of financial symbols.

    This function retrieves price data for the specified symbols over a given time period, 
    computes log returns, and returns either a styled correlation matrix (for 3+ symbols) 
    or a single correlation coefficient (for 2 symbols).

    Parameters:
        syms (list): A list of financial symbols (e.g., stock tickers) to analyze.
        start (int, optional): The number of days of historical data to retrieve, 
            starting from the current date. Defaults to 60.

    Returns:
        pandas.io.formats.style.Styler or float:
            - If len(syms) > 2: A styled pandas DataFrame with the correlation matrix,
              formatted with a 'Greys' background gradient and values rounded to 2 decimals.
            - If len(syms) == 2: A float representing the correlation coefficient between
              the two symbols.

    Notes:
        - Requires `fmp_priceLoop` (assumed to fetch price data) and `utils.ddelt` 
          (assumed to compute a date offset) to be defined elsewhere.
        - Assumes price data in `df` is a pandas DataFrame with symbols as columns and
          dates as rows.
        - Log returns are calculated as the difference of the natural logarithm of prices,
          with the first row (NaN) dropped.
        - For a single symbol (len(syms) == 1), the behavior is undefined and may raise an error.

    Examples:
        >>> fmp_cormatrix(['AAPL', 'MSFT', 'GOOG'])
        # Returns a styled correlation matrix
        >>> fmp_cormatrix(['AAPL', 'MSFT'])
        0.75  # Example correlation coefficient
    """
    df = fmp_priceLoop(syms, start=utils.ddelt(start + 2))
    log_returns_df = np.log(df).diff().dropna()

    if len(syms) > 2:
        # Compute the correlation matrix using pandas
        corr_matrix = pd.DataFrame(log_returns_df.corr())
        styled_df = corr_matrix.style.background_gradient(cmap='Greys').format('{:.2f}')
        return styled_df
    else:
        return log_returns_df.corr().iloc[0, 1]
    
    
#------------------------------------------------------------------------------    


    
#---------------------------------------------------------------------------------
    
## MOD 2/6/26 7:56 am
# 'apikey' is assumed to be assigned at the module level

def fmp_filings(sym, filing_type=None, days_back=None, limit=100):
    """
    Fetches SEC filings for a given symbol.
    
    Parameters:
    - sym (str): Ticker symbol.
    - filing_type (str): Specific form (e.g., '10-k', '10-q', '8-k', '4').
    - days_back (int): Filter for filings within the last N days.
    - limit (int): Max filings to fetch from the API.
    
    Common codes for 'filing_type':
    - '10-k'   : Annual report
    - '10-q'   : Quarterly report
    - '8-k'    : Material events/Earnings releases
    - '4'      : Insider transactions
    - 'def 14a': Proxy statements (exec pay)
    """
    # Normalize type to lowercase as requested
    f_type = filing_type.lower() if filing_type else None
    
    url = f"https://financialmodelingprep.com/api/v3/sec_filings/{sym}?page=0&limit={limit}&apikey={apikey}"
    if f_type:
        url += f"&type={f_type}"
        
    df = pd.read_json(url)
    
    if df.empty:
        return None
        
    # Standardize columns and convert date
    df = df[['acceptedDate', 'cik', 'type', 'link']]
    df['acceptedDate'] = pd.to_datetime(df['acceptedDate'])
    
    # Apply date filter at the DataFrame level
    if days_back:
        threshold = datetime.now() - timedelta(days=days_back)
        df = df[df['acceptedDate'] >= threshold]
        
    if df.empty:
        print(f"No filings found for {sym} in the last {days_back} days.")
        return None

    def make_clickable(val):
        return f'<a target="_blank" href="{val}">Link</a>'
    
    return df.style.format({'link': make_clickable})

# Usage: Get all 8-K filings from the last 30 days
# fmp_filings('AMKR', filing_type='8-k', days_back=30)
    
#-----------------------------------------------------------------------------------------

def fmp_prof(syms, facs=['companyName','sector', 'industry', 'mktCap'] ):
    '''   
    returns dataframe for a symbol or list of symbols
    facs = facs = 'symbol', 'price', 'beta', 'volAvg', 'mktCap', 'lastDiv', 'range', 
       'changes', 'companyName', 'currency', 'cik', 'isin', 'cusip', 'exchange', 
       'exchangeShortName', 'industry', 'website', 'description', 'ceo', 'sector', 
       'country', 'fullTimeEmployees', 'phone', 'address', 'city', 'state', 'zip', 
       'dcfDiff', 'dcf', 'image', 'ipoDate', 'defaultImage', 'isEtf', 
       'isActivelyTrading' 
        '''
    
    if isinstance(syms,str):
        syms=syms
    else:    
        syms=tuple(syms)
        syms=','.join(syms)
    profurl=requote_uri('https://financialmodelingprep.com/api/v3/profile/'+syms+'?apikey='+apikey)
    response = urlopen(profurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    idx = [sub['symbol'] for sub in stuff]

    #facs=['companyName', 'beta', 'volAvg', 'mktCap', 'cik']
    df = pd.DataFrame([[sub[k] for k in facs ] for sub in stuff], 
                  columns= facs, index=idx)
    if 'volAvg' in facs:
        df.volAvg = np.round(df.volAvg/1000000,0)
    if 'mktCap' in facs:
        df.mktCap = np.round(df.mktCap/1000000,0)
   
    return df

#-------------------------------------------------------

def fmp_profF(sym, facs=None ):
    '''  
       Returns the full profile of a single symbol
       facs = 'symbol', 'price', 'beta', 'volAvg', 'mktCap', 'lastDiv', 'range', 
       'changes', 'companyName', 'currency', 'cik', 'isin', 'cusip', 'exchange', 
       'exchangeShortName', 'industry', 'website', 'description', 'ceo', 'sector', 
       'country', 'fullTimeEmployees', 'phone', 'address', 'city', 'state', 'zip', 
       'dcfDiff', 'dcf', 'image', 'ipoDate', 'defaultImage', 'isEtf', 
       'isActivelyTrading'   '''
    if facs==None:
        full=['symbol', 'companyName', 'price', 'beta', 'volAvg', 'mktCap', 'lastDiv',  
			'changes', 'companyName', 'currency', 'cik', 'isin', 'cusip',  
			'exchangeShortName', 'industry',  'sector', 'country', 
			'ipoDate', 'isEtf', 'isActivelyTrading' , 'description'  ]	
        facs=full
    profurl=requote_uri('https://financialmodelingprep.com/api/v3/profile/'+sym+'?apikey='+apikey)
    response = urlopen(profurl, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)
    try:
        stuff= stuff[0]
    except IndexError:
        d=dict.fromkeys(facs)
        d.update(symbol=sym)
        return d
    return dict((k, stuff[k]) for k in facs)
#---------------------------------------------------------------------------------------------------

######Check monthly with return flag=True
def fmp_2SymReg(sym_a, sym_b, start='1960-01-01', end=str(dt.datetime.now().date()), ret=True):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # Retrieve data once
    data = fmp_priceLoop([sym_a, sym_b], start=start, end=end, fac='close').dropna()

    # Daily regression
    if ret:
        log_returns = np.log(data / data.shift()).dropna()
        X_daily = log_returns.iloc[:, 0].to_numpy().reshape(-1, 1)
        Y_daily = log_returns.iloc[:, 1].to_numpy().reshape(-1, 1)
    else:
        X_daily = data.iloc[:, 0].to_numpy().reshape(-1, 1)
        Y_daily = data.iloc[:, 1].to_numpy().reshape(-1, 1)
    
    lin_regr = LinearRegression()
    lin_regr.fit(X_daily, Y_daily)
    Y_pred_daily = lin_regr.predict(X_daily)

    alpha_daily = lin_regr.intercept_[0]
    beta_daily = lin_regr.coef_[0, 0]
    r_squared_daily = np.round(r2_score(Y_daily, Y_pred_daily), 3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Daily:  Alpha: " + str(round(alpha_daily, 5)) + ", Beta: " + str(round(beta_daily, 3)) + "  R²: " + str(r_squared_daily))
    ax.scatter(X_daily, Y_daily)
    ax.plot(X_daily, Y_pred_daily, c='r')
    ax.set_xlabel(sym_a)
    ax.set_ylabel(sym_b)

    # Monthly regression
    data_monthly = data.resample('M').last().dropna()
    if ret:
        log_returns_monthly = np.log(data_monthly / data_monthly.shift()).dropna()
        X_monthly = log_returns_monthly.iloc[:, 0].to_numpy().reshape(-1, 1)
        Y_monthly = log_returns_monthly.iloc[:, 1].to_numpy().reshape(-1, 1)
    else:
        X_monthly = data_monthly.iloc[:, 0].to_numpy().reshape(-1, 1)
        Y_monthly = data_monthly.iloc[:, 1].to_numpy().reshape(-1, 1)

    lin_regr.fit(X_monthly, Y_monthly)
    Y_pred_monthly = lin_regr.predict(X_monthly)

    alpha_monthly = lin_regr.intercept_[0]
    beta_monthly = lin_regr.coef_[0, 0]
    r_squared_monthly = np.round(r2_score(Y_monthly, Y_pred_monthly), 3)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Monthly:  Alpha: " + str(round(alpha_monthly, 5)) + ", Beta: " + str(round(beta_monthly, 3)) + "  R²: " + str(r_squared_monthly))
    ax.scatter(X_monthly, Y_monthly)
    ax.plot(X_monthly, Y_pred_monthly, c='r')
    ax.set_xlabel(sym_a)
    ax.set_ylabel(sym_b)
#-----------------------------------------------------------------------------------------------------

def fmp_stoch(sym,length=8, smooth=3):
    df=fmp_price(sym, facs=['low', 'high', 'close'], start=tdelt(length+3))
    df['highest'] = df.high.rolling(length).max()
    df['lowest'] = df.low.rolling(length).min()
    df['k'] = 100*(df.close-df.lowest) / (df.highest-df.lowest)
    df['k_smooth'] = df.k.rolling(smooth).mean()
    return np.round(df.k_smooth[-1],2)

#------------------------------------------------------------------------------------------------------
def fmp_rsi(sym, periods = 3, watch=True, start='1990-01-01'):
    """
    Returns a pd.Series with the relative strength index.
    """
    if watch:
        df=fmp_price(sym, facs=['close'], start=utils.ddelt(periods+5))
    
        close_delta = df.diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
    

        # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
 
        
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
       

        return np.round(rsi.close[-1],2)
    
    else:
        df=fmp_price(sym, facs=['close'], start=start)
    
        close_delta = df.diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
    

        # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
 
        
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
       

        return np.round(rsi.close,2)
    
#-----------------------------------------------------------------------------------

def fmp_peers(sym):
    '''
    Input: Symbol or list of symbols
    Returns:  sym plus a list of peer symbols
    '''
    sym=sym.upper()
    insurl=f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={sym}&apikey={apikey}"
    response = urlopen(insurl)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)

    lst = stuff[0]['peersList']
    lst.insert(0,sym)
    df=pd.DataFrame([fmp_profF(i, facs=['companyName', 'mktCap', 'sector', 'industry', 
                                        'country', 'exchangeShortName']) for i in lst], index=lst)
    df['mktCap']=(df.mktCap/1000000).round(2)
    df.rename(columns={'mktCap': 'mktCap(M)'}, inplace=True)
    df.sort_values(by='mktCap(M)', ascending=False, inplace=True)
    return df

#---------------------------------------------------------------------------

def fmp_plotFin(data, title=None):
    ''' parameters are data and title'''
    
    if title==None:
        title = data.columns[0]
    else:
        title = str(title)
    ddata=data.iloc[:,0]    
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "lightgray",
        "text.color": "orange",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "orange",
        "xtick.color": "lightgray",
        "xtick.labelsize":16,
        "ytick.color": "lightgray",
        "ytick.labelsize":16,
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "lightgray",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})
    fig,ax= plt.subplots(figsize=(14, 8))    
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    plt.plot(data, color='limegreen')
    plt.title(title, loc='left',fontdict={'family': 'serif', 
                    'color' : 'orange',
                    'weight': 'bold',
                    'size': 20})
    ax.grid(linestyle=':')
    ax.tick_params(axis='x', labelrotation=45)
    plt.show()
    
#------------------------------------------------------------------------
def fmp_plotFinMult(data, title='Multi-Symbol Chart'):
    ''' parameters are data and title'''
    
    
    title = str(title)
    plt.rcParams.update({
        "lines.color": "white",
        "patch.edgecolor": "lightgray",
        "text.color": "orange",
        "axes.facecolor": "black",
        "axes.edgecolor": "lightgray",
        "axes.labelcolor": "orange",
        "xtick.color": "lightgray",
        "xtick.labelsize":16,
        "ytick.color": "lightgray",
        "ytick.labelsize":16,
        "grid.color": "lightgray",
        "figure.facecolor": "black",
        "figure.edgecolor": "lightgray",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})
    fig,ax= plt.subplots(figsize=(14, 8))    
    ax.yaxis.tick_right()
    plt.plot(data)
    plt.legend(data.columns, loc='lower right')
    plt.title(title, loc='left',fontdict={'family': 'serif', 
                    'color' : 'orange',
                    'weight': 'bold',
                    'size': 20})
    ax.grid(linestyle=':')
    plt.show()	
    
##------------------------------------------------

def fmp_plotBarRetts(data): 
    """
    Plots a bar chart of returns over time, converting dates to categorical labels
    to avoid gaps for non-trading days. Automatically adjusts the number of x-axis ticks.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame with a datetime index and a single column of return values.
    
    Returns:
    --------
    None
        Displays the plotted bar chart.
    """
    # Drop NaN values
    data = data.dropna()

    # Extract values
    values = data.iloc[:, 0].to_numpy().flatten()
    colors = np.where(values >= 0, 'g', 'r')  # Assign green for positive, red for negative

    # Convert dates to string labels (so they are treated as categorical)
    date_labels = data.index.strftime('%m/%d/%Y')

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid()

    # Use range as x-values to plot without gaps
    ax.bar(range(len(data)), values, color=colors)

    # Determine tick frequency based on number of data points
    n = len(data)
    if n <= 10:
        tick_freq = 1
    elif n <= 30:
        tick_freq = 2
    elif n <= 60:
        tick_freq = 5
    elif n <= 120:
        tick_freq = 10
    else:
        tick_freq = max(n // 15, 1)

    # Set ticks and labels
    ticks = range(0, len(data), tick_freq)
    ax.set_xticks(ticks)
    ax.set_xticklabels([date_labels[i] for i in ticks], rotation=45)

    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd


def fmp_plotDualAxis(df, left_cols=None, right_cols=None, left_label=None, right_label=None):
    """
Plots the given DataFrame using dual y-axes.

:param df: The DataFrame to plot. 
:param left_cols: A column name or a list of column names as strings to plot on the left y-axis.
:param right_cols: A column or a list of column names ast strings to plot on the right y-axis.
:param left_label: The label for the left y-axis as a string.
:param right_label: The label for the right y-axis as a string.
    """

    if left_cols is None and right_cols is None:
        raise ValueError("At least one of left_cols or right_cols must be specified")

    fig, ax1 = plt.subplots(figsize=(12,7))
    ax2 = ax1.twinx()

    if left_cols:
        if isinstance(left_cols, str):
            left_cols = [left_cols]
        for i, col in enumerate(left_cols):
            color = None if i > 0 else 'C0'
           
            if col in df.columns:
                if i == 0:
                    ax1.plot(df.index, df[col], label=left_label or col, color=color, linewidth=3.0)
                else:
                    ax1.plot(df.index, df[col], label=left_label or col, color=color)
        ax1.set_ylabel(left_label or ', '.join(left_cols), color='black')
        ax1.spines['left'].set_color('black')

    if right_cols:
        if isinstance(right_cols, str):
            right_cols = [right_cols]
        for i, col in enumerate(right_cols):
            color = plt.cm.tab10(i+len(left_cols)) if left_cols else plt.cm.tab10(i)
            if col in df.columns:
                ax2.plot(df.index, df[col], label=right_label or col, color=color)
        ax2.set_ylabel(right_label or ', '.join(right_cols), color='black')
        ax2.spines['right'].set_color('black')
        ax2.tick_params(axis='y', labelcolor='black')

    # set legend for left axis
    if left_cols:
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc="upper left")

    # set legend for right axis
    if right_cols:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines2, labels2, loc="upper right")

    # remove xlabel if no data for left axis
    if not left_cols:
        ax1.set_xlabel('')

    # remove ylabel if no data for left axis
    if not right_cols:
        ax2.set_ylabel('')

    # add gridlines if data is present
    if left_cols and right_cols:
        ax1.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)
    elif right_cols:
        ax2.grid(True, linestyle='--', linewidth=0.5, color='grey', alpha=0.5)

#---------------------------------------------------------------

def fmp_plotStackedRet(retSyms, lineSym, start=None):
    '''
This function plots 3 stacked Return plots over a Line plot.

Inputs:
    retSyms is a list of exactly 3 symbols.  default is 'SPY', 'QQQ', 'IWM'
    lineSym is a string of a single symbol. defaul is '^TNX' (10Y Treas)
    start is a string 'YYYY-mm-dd'. default is 3 Months
Returns:
    4 vertically stacked plots with the first three being vertical bar plots of price returns
    and the last being a line plot of price
    
    
    
    '''
    
    lineSym=[lineSym]
    syms=retSyms+lineSym

    import pandas as pd
    import matplotlib.pyplot as plt


    df=fmp_priceMult(syms,start=start)

    df.iloc[:, :3] = df.iloc[:, :3].pct_change()
    df=df.dropna()



    # create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)

    # create the bar plots for the top 3 subplots
    for i, col in enumerate(df.columns[:3]):
        axs[i].bar(range(len(df)), df[col], color=['green' if val>0 else 'red' for val in df[col]])
        axs[i].set_ylabel(col)
        axs[i].grid(True)
        if i == 0:
            axs[i].set_title('Returns Over '+df.columns[3])  # Add title to the first plot

    # create the line plot for the bottom subplot
    axs[3].plot(range(len(df)), df.iloc[:, 3], 'k-', linewidth=2)
    axs[3].set_ylabel(df.columns[3])
    axs[3].grid(True)

    # set the tick positions and labels
    first_date = df.index[0]
    last_date = df.index[-1]
    tick_positions = np.linspace(df.index.get_loc(first_date), df.index.get_loc(last_date), num=10, dtype=int)
    tick_labels = df.index[tick_positions].strftime('%Y-%m-%d')

    # set the tick positions and labels for the x-axis
    plt.xticks(tick_positions, tick_labels, rotation=45)

    # adjust subplot spacing
    plt.subplots_adjust(hspace=0.1)

    # save the figure as a PDF file
    #plt.savefig('my_plot.pdf', bbox_inches='tight')


#---------------------------------------------------------------

# Ensure apikey is defined globally in your file
# apikey = '...' 

def build_sector_industry_map():
    """
    Retrieves all stocks and builds a Sector -> Industry map in ONE request.
    Relies on the global 'apikey' variable.
    """
    url = f"https://financialmodelingprep.com/api/v3/stock-screener?limit=10000&apikey={apikey}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        sector_map = defaultdict(set)
        
        for item in data:
            sec = item.get('sector')
            ind = item.get('industry')
            if sec and ind:
                sector_map[sec].add(ind)
        
        return {k: sorted(list(v)) for k, v in sector_map.items()}

    except requests.exceptions.RequestException as e:
        print(f"API Request failed: {e}")
        return {}

def fmp_sectInd(sector=None):
    """
    Display the industries for a specific sector, or all sectors if left blank.

    Parameters:
    -----------
    sector : str, optional
        The specific sector to filter by (Case Sensitive). 
        If None, displays all sectors and industries.

    Available Sectors:
    ------------------
    - Basic Materials
    - Communication Services
    - Consumer Cyclical
    - Consumer Defensive
    - Energy
    - Financial Services
    - Healthcare
    - Industrials
    - Real Estate
    - Technology
    - Utilities
    """
    sector_industry_map = build_sector_industry_map()

    if not sector_industry_map:
        print("No data found.")
        return

    # If a specific sector is requested, filter the map
    if sector:
        # Check if the sector exists in our map
        if sector in sector_industry_map:
            display(Markdown(f"**{sector} Industries:**"))
            for industry in sector_industry_map[sector]:
                display(Markdown(f"- {industry}"))
        else:
            print(f"Sector '{sector}' not found. Please check the docstring for valid sector names.")
            
    # If no sector is provided, show everything (default behavior)
    else:
        for sec in sorted(sector_industry_map.keys()):
            industries = sector_industry_map[sec]
            display(Markdown(f"**{sec}:**"))
            for industry in industries:
                display(Markdown(f"- {industry}"))

#----------------------------------------------------------------------------------------
def fmp_plotMult(sym = 'COIN', start=None):
    df=fmp_price(sym, start=start)
    df['rsi'] = fmp_rsi(sym,periods=3, watch=False)

    df['max'] = df.close.rolling(42).max()
    df['min'] = df.close.rolling(42).min()
    df['mid'] = (df['max'] + df['min'])/2


    df['indx']=(df['close'] - df['min']) / (df['max'] - df['min'])

    df['width'] = (df['max'] - df['min']) / df['close']

    max_val = df['close'].expanding().max()

    # create a new column 'max_col' with the maximum value repeated for each row
    df['max_col'] = max_val

    df['dd'] = (df['close']-df['max_col']) / df['max_col']

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True , gridspec_kw={'height_ratios': [4, 1, 1, 2,1]}, figsize=(12,13))

    # plot the 'close' column on the first subplot
    df[['close', 'max', 'min', 'mid']].plot(ax=ax1, grid=True, color = ['blue', 'gray', 'gray', 'orange'])
    ax1.legend().set_visible(False)
    ax1.set_title(sym)

    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position('right')

    # plot the 'volume' column on the second subplot
    df['indx'].plot(ax=ax2, grid=True)
    ax2.legend()
    ax2.axhline(y=.5, color='red', linestyle='--')

    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')

    # plot the 'volume' column on the second subplot
    df['width'].plot(ax=ax3, grid=True)
    ax3.legend()

    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')

    # plot the 'volume' column on the second subplot
    df['rsi'].plot(ax=ax4, grid=True)
    ax4.legend()

    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')

    ax4.axhline(y=85, color='red', linestyle='--')
    ax4.axhline(y=15, color='red', linestyle='--')


    # plot the 'volume' column on the second subplot
    df['dd'].plot(ax=ax5, grid=True)
    ax5.fill_between(df.index, 0,df['dd'], color='blue', alpha=0.2)
    ax5.legend()

    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_position('right')

#------------------------------------------------------------------------------

def fmp_plotPiv(sym='SPY', start=None):
    df=fmp_price(sym, start=start)

    window=30
    volLength=21
    # Calculate log returns 
    log_returns = np.log(df / df.shift())

    # Calculate rolling window standard deviation
    rolling_std = log_returns.rolling(window).std()

    # Multiply by square root of volLength
    result = rolling_std * np.sqrt(volLength)

    result = result.fillna(method='bfill')
    result = result.rename(columns={'close': 'hv'})

    PEAK, VALLEY = 1, -1

    def _identify_initial_pivot(X, init_thresh):
        """Quickly identify the X[0] as a peak or valley."""
        x_0 = X[0]
        max_x = x_0
        max_t = 0
        min_x = x_0
        min_t = 0
        up_thresh = 1
        down_thresh = 1

        for t in range(1, len(X)):
            x_t = X[t]

            if x_t / min_x >= init_thresh:
                return VALLEY if min_t == 0 else PEAK

            if x_t / max_x <= -init_thresh:
                return PEAK if max_t == 0 else VALLEY

            if x_t > max_x:
                max_x = x_t
                max_t = t

            if x_t < min_x:
                min_x = x_t
                min_t = t

        t_n = len(X)-1
        return VALLEY if x_0 < X[t_n] else PEAK



    initial_pivot=_identify_initial_pivot(df.close, init_thresh=result.hv[0])



    ##--------------------------------------------------------------------------------------------
    close = df.close


    """
    Finds the peaks and valleys of a series of HLC (open is not necessary).
    TR: This is modified peak_valley_pivots function in order to find peaks and valleys for OHLC.
    Parameters
    ----------
    close : This is series with closes prices.
    up_thresh : The minimum relative change necessary to define a peak.
    down_thesh : The minimum relative change necessary to define a valley.
    Returns
    -------
    an array with 0 indicating no pivot and -1 and 1 indicating valley and peak
    respectively
    Using Pandas
    ------------
    For the most part, close may be a pandas series. However, the index must
    either be [0,n) or a DateTimeIndex. Why? This function does X[t] to access
    each element where t is in [0,n).
    The First and Last Elements
    ---------------------------
    The first and last elements are guaranteed to be annotated as peak or
    valley even if the segments formed do not have the necessary relative
    changes. This is a tradeoff between technical correctness and the
    propensity to make mistakes in data analysis. The possible mistake is
    ignoring data outside the fully realized segments, which may bias analysis.
    """

    up_thresh = result.hv
    down_thresh = -result.hv

    if down_thresh[0] > 0:
        raise ValueError('The down_thresh must be negative.')


    t_n = len(close)
    pivots = np.zeros(t_n, dtype='i1')
    pivots[0] = initial_pivot

    # Adding one to the relative change thresholds saves operations. Instead
    # of computing relative change at each point as x_j / x_i - 1, it is
    # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
    # This saves (t_n - 1) subtractions.
    up_thresh += 1
    down_thresh += 1

    trend = -initial_pivot

    last_pivot_t = 0
    last_pivot_x = close[0]
    for t in range(1, len(close)):

        if trend == -1:
            x = close[t]

            r = x / last_pivot_x


            if r >= up_thresh[t]:
                pivots[last_pivot_t] = trend#
                trend = 1
                #last_pivot_x = x
                last_pivot_x = close[t]
                last_pivot_t = t
            elif x < last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t
        else:
            x = close[t]

            r = x / last_pivot_x

            if r <= down_thresh[t]:
                pivots[last_pivot_t] = trend
                trend = -1
                #last_pivot_x = x
                last_pivot_x = close[t]
                last_pivot_t = t
            elif x > last_pivot_x:
                last_pivot_x = x
                last_pivot_t = t


    if last_pivot_t == t_n-1:
        pivots[last_pivot_t] = trend
    elif pivots[t_n-1] == 0:
        pivots[t_n-1] = trend
    ar= pd.DataFrame(pivots, index=df.index, columns=['pivot'])    
    df=pd.concat([df,ar, result], axis=1)

    df['count'] = range(1, len(df)+1)

    dfp=df[df['pivot']!=0]


    dfp['dur'] = dfp['count'].diff().fillna(0).astype(int)

    dfp['chg']=np.round(dfp.close.pct_change(),3)

    dfp['ave'] = np.round(dfp.chg / dfp.dur,4)


    conditions = [
        (dfp['pivot'] == 1) & (dfp['close'] > dfp['close'].shift(2)),
        (dfp['pivot'] == 1) & (dfp['close'] < dfp['close'].shift(2)),
        (dfp['pivot'] == -1) & (dfp['close'] > dfp['close'].shift(2)),
        (dfp['pivot'] == -1) & (dfp['close'] < dfp['close'].shift(2))
    ]

    choices = ['HH', 'LH', 'HL', 'LL']

    dfp['pivot2'] = np.select(conditions, choices, default=np.nan)

    # create two subsets of the data based on the pivot column
    subset_up = dfp[dfp['pivot'] > 0][['pivot2', 'dur', 'chg']]
    subset_dn = dfp[dfp['pivot'] < 0][['pivot2', 'dur', 'chg']]

    # get the pivot points for each subset
    pivot_up = dfp[dfp['pivot'] > 0]['close'].tolist()
    pivot_dn = dfp[dfp['pivot'] < 0]['close'].tolist()

    # plot the DataFrame and pivot points
    fig, ax = plt.subplots(figsize=(15,9))

    df['close'].plot(ax=ax)

    ax.plot(df.loc[df['pivot'] != 0, 'close'], '--', color='r')

    # annotate the first subset
    for i in range(len(subset_up)):
        text = f"{  subset_up.iloc[i]['pivot2']}\n{subset_up.iloc[i]['dur']}\n{subset_up.iloc[i]['chg']}"
        ax.annotate(text, xy=(subset_up.index[i], pivot_up[i]), xytext=(-20, 15), textcoords='offset pixels')

    # annotate the second subset    
    for i in range(len(subset_dn)):
        text = f"{  subset_dn.iloc[i]['pivot2']}\n{subset_dn.iloc[i]['dur']}\n{subset_dn.iloc[i]['chg']}"
        ax.annotate(text, xy=(subset_dn.index[i], pivot_dn[i]), xytext=(-20, -45), textcoords='offset pixels')

    # set axis labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(sym)
    ax.grid()
    plt.show()

#------------------------------------------------------------------------------------------
def fmp_newsdict(sym=None, limit='50'):
    '''
    syms= symbols in the form of: 'KSHB,GNLN,PBL,NBN,SKT' .  Max 5 symbol limit by FMP
    convert list of strings by: ','.join(['KSHB', 'GNLN', 'PBL'])
    returns a dict
    '''
    
    if sym:
        sym=sym.upper()
        symnewsurl = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={sym}&limit={limit}&apikey={apikey}"
        response = urlopen(symnewsurl, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff=json.loads(data)
        
        
            
    else:
        allnewsurl = f"https://financialmodelingprep.com/api/v3/stock_news?limit={limit}&apikey={apikey}"
        response = urlopen(allnewsurl, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff=json.loads(data)
        
    return stuff    
        
#--------------------------------
	
def fmp_news(sym=None, limit='50'):
    '''
    Retrieve and display stock news from Financial Modeling Prep API.

    Parameters:
    -----------
    sym : str, optional
        Stock ticker symbol(s) to query. Can be a single symbol (e.g., 'AAPL') 
        or multiple symbols as a comma-separated string (e.g., 'AAPL,MSFT,GOOGL').
        If None, returns general stock news not specific to any symbol.
        Default is None.
    limit : str, optional
        Maximum number of news articles to retrieve, as a string.
        Default is '50'.

    Returns:
    --------
    None
        Displays formatted HTML output in Jupyter notebook showing:
        - Stock symbol and publication date (bold)
        - News source site (italicized)
        - News text content
        - Clickable URL to original article

    Examples:
    ---------
    >>> fmp_news('AAPL')  # Get news for Apple
    >>> fmp_news('AAPL,MSFT', limit='25')  # Get 25 news items for Apple and Microsoft
    >>> fmp_news()  # Get general stock market news
    
    Notes:
    ------
    - Requires an API key stored in variable 'apikey'
    - Uses Financial Modeling Prep API (https://financialmodelingprep.com/api/v3/stock_news)
    - Must be run in a Jupyter notebook environment for HTML display
    - Requires internet connection and imported dependencies: urlopen, certifi, json
    - Symbols are automatically converted to uppercase
    
    Raises:
    -------
    Depends on API response - may raise exceptions if:
        - API key is invalid
        - Network connection fails
        - Invalid ticker symbols provided
    '''
    
    if sym:
        sym = sym.upper()
        symnewsurl = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={sym}&limit={limit}&apikey={apikey}"
        response = urlopen(symnewsurl, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff = json.loads(data)
        for i in stuff:
            display(HTML(f"<p style='font-size:20px; font-weight:bold;'>{i['symbol']} - {i['publishedDate']}</p>"))
            display(HTML(f"<p style='font-style:italic;'>{i['site']}</p>"))
            display(HTML(f"<p>{i['text']}</p>"))
            display(HTML(f"<a href='{i['url']}' target='_blank'>{i['url']}</a>"))
            display(HTML("<br>"))
            
    else:
        allnewsurl = f"https://financialmodelingprep.com/api/v3/stock_news?limit={limit}&apikey={apikey}"
        response = urlopen(allnewsurl, context=ssl_context)
        data = response.read().decode("utf-8")
        stuff = json.loads(data)
        for i in stuff:
            display(HTML(f"<p style='font-size:20px; font-weight:bold;'>{i['symbol']} - {i['publishedDate']}</p>"))
            display(HTML(f"<p style='font-style:italic;'>{i['site']}</p>"))
            display(HTML(f"<p>{i['text']}</p>"))
            display(HTML(f"<a href='{i['url']}' target='_blank'>{i['url']}</a>"))
            display(HTML("<br>"))
#-------------------------------------------------------
def fmp_divHist(sym):
    '''
    input: symbol as string
    returns:  DataFrame of dividend $/share with ex-date as the index
    '''
    url= f"https://financialmodelingprep.com/api/v3/historical-price-full/stock_dividend/{sym}?apikey={apikey}"
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff=json.loads(data)['historical']
    
    df = pd.DataFrame(stuff)
    df=df.set_index('date')['dividend'].to_frame().sort_index(ascending=True)
    df.index = pd.to_datetime(df.index)
    return df

#-------------------------------------------------------

def fmp_perfStats(s):
    '''
    input: Series or Datframe of prices
    outputs: ab object displaying performance stats
    uses ffn package
    '''
    stats = ffn.calc_stats(s)

# Display performance metrics
    stats.display()
#----------------------------------------------------------

def fmp_ratios(symbol, facs=['currentRatio', 'quickRatio', 'cashRatio',
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
       'enterpriseValueMultiple', 'priceFairValue']):
    '''
       factors=['currentRatio', 'quickRatio', 'cashRatio',
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
    '''
    
  
    facs=['date']+facs
    url=f'https://financialmodelingprep.com/api/v3/ratios/{symbol}?period=quarter&apikey={apikey}'
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
 
    df=pd.DataFrame([{key: value for key, value in item.items() if key in facs} for item in stuff])
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    return df.sort_index()
    
#--------------------------------------------------------------------------

def fmp_ratiosttm(symbol, facs=['dividendYielTTM','dividendYielPercentageTTM','peRatioTTM',
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
       'priceSalesRatioTTM','enterpriseValueMultipleTTM','priceFairValueTTM','dividendPerShareTTM']):
    '''
       facs=['dividendYielTTM','dividendYielPercentageTTM','peRatioTTM',
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
    '''
    
  
    
    url=f'https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}?period=quarter&apikey={apikey}'
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
    stuff = stuff[0]
    
    
    return pd.Series({key: value for key, value in stuff.items() if key in facs}).T

#--------------------------------------------------------------
def fmp_keyMetrics(symbol, facs=['revenuePerShare','netIncomePerShare','operatingCashFlowPerShare',
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
             'roe','capexPerShare']):
    '''
       facs=['revenuePerShare','netIncomePerShare','operatingCashFlowPerShare',
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
    '''
    
  
    facs=['date']+facs
    url=f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period=quarter&apikey={apikey}'
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
    df=pd.DataFrame([{key: value for key, value in item.items() if key in facs} for item in stuff])
    df['date'] = pd.to_datetime(df.date)
    df.set_index('date', inplace=True)
    return df.sort_index()
 
#-------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------

def fmp_earnEst(sym, period='annual'):
    '''
    input:  sym:  as string
            period: as string 'annual' or 'quarter'
    returns: DatFrame wiyh columns  'RevLow','RevHigh','RevAvg','EbitdaLow','EbitdaHigh','EbitdaAvg','EbitLow', 'EbitHigh',
               'EbitAvg','NetIncLow','NetIncHigh','NetIncAvg','SgaExpLow','SgaExpHigh','SgaExpAvg',
            'EpsAvg','EpsHigh','EpsLow','numRev','numEps'        
    '''
    url=f"https://financialmodelingprep.com/api/v3/analyst-estimates/{sym}?&period={period}&apikey={apikey}"
    df=pd.read_json(url)
    df.set_index('date', inplace=True)
    abbreviations = [
    'sym',
    'RevLow',
    'RevHigh',
    'RevAvg',
    'EbitdaLow',
    'EbitdaHigh',
    'EbitdaAvg',
    'EbitLow',
    'EbitHigh',
    'EbitAvg',
    'NetIncLow',
    'NetIncHigh',
    'NetIncAvg',
    'SgaExpLow',
    'SgaExpHigh',
    'SgaExpAvg',
    'EpsAvg',
    'EpsHigh',
    'EpsLow',
    'numRev',
    'numEps'
]
    df.columns=abbreviations
    return df.sort_index()
#--------------------------------------------------------


def fmp_plotyc():
    # Initialize TvDatafeed
    tv = TvDatafeed()
    
    xlst = ['1M','3M','6M','1Y','2Y','3Y','5Y','7Y','10Y','20Y','30Y']
    symlst = ['US01MY', 'US03MY', 'US06MY', 'US01Y', 'US02Y', 'US03Y', 'US05Y', 'US07Y', 'US10Y', 'US20Y', 'US30Y']
    
    # Initialize empty DataFrame
    df = pd.DataFrame()
    
    # Collect historical data
    for i in symlst:
        data = tv.get_hist(i, 'TVC', n_bars=260)
        if data is not None and 'close' in data.columns:
            df[i] = data['close']
    
    # Check if df is empty before proceeding
    if df.empty:
        raise ValueError("DataFrame is empty. Check if 'tv.get_hist' is returning data.")
    
    # Ensure lookback indices are within DataFrame bounds
    lookback = [1, 130, -21, -2]
    df_length = len(df)
    valid_lookback = [idx for idx in lookback if -df_length <= idx < df_length]
    
    # If no valid indices, raise an error
    if not valid_lookback:
        raise IndexError("No valid indices in lookback range.")
    
    # Slice DataFrame with valid lookback indices
    dff = df.iloc[valid_lookback].T
    
    # # Convert column names to dates
    # dff.columns = pd.to_datetime(dff.columns, errors='coerce').strftime('%Y-%m-%d')
    
    # # Drop any columns that could not be converted to dates
    # dff = dff.loc[:, ~dff.columns.str.contains('NaT', na=False)]
    
    # # Debugging: Print lengths and contents
    # print(f"Length of xlst: {len(xlst)}")
    # print(f"Columns in dff: {dff.columns.tolist()}")
    
    # # Ensure length of xlst matches number of dff columns
    # if len(xlst) != len(dff.columns):
    #     raise ValueError("Mismatch between x-axis labels and DataFrame columns.")
    
    # # Plot the data
    # fig, ax = plt.subplots(figsize=(12, 8))
    # for column in dff.columns:
    #     ax.plot(xlst, dff[column], label=column)
    
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    # plt.grid(True, which='both', linestyle=':', linewidth='0.5', color='lightgrey')
    # plt.legend(title='Symbols')
    # plt.show()
    return dff

#-------------------------------------------------------------------------------------------------------------------------

def fmp_plotShYield(sym, output='plot', quarters=40):
    """
    Compute and visualize the Shareholder Yield (SH Yield) for a given stock symbol.
    
    The function retrieves financial data for the specified symbol, calculates the dividend yield,
    buyback yield, and total shareholder yield, and either plots the data or returns it as a DataFrame.
    
    Parameters:
    sym (str): Stock ticker symbol.
    output (str, optional): Determines the function output. Default is 'plot'.
        - 'plot': Displays a plot of the shareholder yield components.
        - 'df': Returns a DataFrame containing the yield calculations.
        - 'quarters': is the number of quarters to plot.  Default is 10 years or 40 quarters
    
    Returns:
    None or pd.DataFrame: 
        - Returns None if output='plot' (displays a plot).
        - Returns a DataFrame if output='df'.
    """
    # Fetch financial data
    cf = fmp_cashfts(sym, facs=['commonStockIssued', 'commonStockRepurchased', 'dividendsPaid'])
    cf=cf.rolling(4).sum()
    mc = fmp_mcap(sym)
    
    # Merge data
    sy = pd.concat([cf, mc], axis=1, join='inner')
    
    # Compute yields
    sy['divYield'] = -sy['dividendsPaid'] / sy['mktCap']
    sy['bbYield'] = (-sy['commonStockRepurchased'] - sy['commonStockIssued']) / sy['mktCap']
    sy['shYield'] = np.round((-sy['dividendsPaid'] - sy['commonStockRepurchased'] - sy['commonStockIssued']) / sy['mktCap'], 4)
    
    if output == 'df':
        return sy
    sy=sy[-quarters:]    
    
    # Plot the shareholder yield components
    plt.figure(figsize=(10, 6))
    plt.plot(sy.index, sy['divYield'], label='Dividend Yield', color='blue')
    plt.plot(sy.index, sy['divYield'] + sy['bbYield'], label='Shareholder Yield', color='green')
    plt.fill_between(sy.index, sy['divYield'], sy['divYield'] + sy['bbYield'], color='green', alpha=0.5)
    plt.fill_between(sy.index, 0, sy['divYield'], color='blue', alpha=0.5)
    plt.grid()
    plt.title(f'Shareholder Yield for {sym}')
    plt.legend()
    plt.show()


#-------------------------------------------------------------------------------------------------------------------------------


def fmp_growth(sym,facs=[ "fiveYOperatingCFGrowthPerShare"]):
    ''' "revenueGrowth",
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
    "sgaexpensesGrowth"'''
    url=f'https://financialmodelingprep.com/api/v3/financial-growth/{sym}?period=annual&apikey={apikey}'
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)
    stuff = stuff[0]
    
    
    return pd.Series({key: value for key, value in stuff.items() if key in facs}).T


#--------------------------------------------------------------

def fmp_isActive(syms):
    """Check if stock symbols are actively trading using FMP profile data.
    
    Args:
        syms (str or list): A single stock symbol as a string or a list of stock symbols.
    
    Returns:
        None: This function does not return a value. It prints the symbol and its status
              if the symbol is not actively trading or not found in the FMP system.
    
    Prints:
        str: For each symbol that is not actively trading:
            - "<symbol> False" if the symbol exists but is not actively trading
            - "<symbol> Possible Bad Symbol" if the symbol is not found or has no status
    
    Example:
        >>> fmp_isActive('AAPL')
        # No output if AAPL is actively trading
        >>> fmp_isActive(['AAPL', 'INVALID', 'XYZ'])
        INVALID Possible Bad Symbol
        XYZ Possible Bad Symbol
        # Assuming INVALID and XYZ are not valid/active symbols
    
    Notes:
        - Requires the `fmp_profF` function to fetch profile data from Financial Modeling Prep.
        - Depends on the FMP API returning a dictionary with 'isActivelyTrading' key.
        
    """
    all_active = True
    
    # Ensure syms is a list, even if a single string is provided
    if isinstance(syms, str):
        syms = [syms]

    for sym in syms:
        status = fmp_profF(sym).get('isActivelyTrading')   
        if status is not True:
            all_active = False  # Set flag to False if any symbol is inactive or bad
            if status is None:
                print(sym, 'Possible Bad Symbol')
            else:
                print(sym, status)

    if all_active:
        print("All Symbols are Active on the FMP System")

#------------------------------------------------------------------------------------------------
def fmp_isin(isin):
    url = f"https://financialmodelingprep.com/api/v4/search/isin?isin={isin}&apikey="+apikey
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    return json.loads(data)
    
#-----------------------------------------------------------

def fmp_transcript(sym, year=None, quarter=None, output=None):
    '''
    Retrieves earning call transcripts with a structured header.
    - If year and quarter provided: Returns that specific transcript.
    - If not provided: Returns the most recent transcript available.
    - output=None: Returns LLM-optimized text (unwrapped) - default.
    - output='print': Prints formatted text (wrapped) to screen.
    - output='file': Saves formatted text (wrapped) to '[sym]_[year]_Q[quarter].txt'.
    - output='list': Returns DataFrame of all available transcripts.
    '''
    sym = sym.upper()

    # Build URL - v3 endpoint works with or without year/quarter
    if year and quarter:
        url = f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{sym}?quarter={quarter}&year={year}&apikey={apikey}'
    else:
        # Without year/quarter, returns all transcripts (most recent first)
        url = f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{sym}?apikey={apikey}'

    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)

    if not stuff:
        print(f"No transcripts found for {sym}.")
        return None

    # If user wants the list of all available transcripts
    if output == 'list':
        return pd.DataFrame(stuff)

    # Get the first (most recent) transcript
    transcript_data = stuff[0]
    year = transcript_data.get('year')
    quarter = transcript_data.get('quarter')
    date_time = transcript_data.get('date', 'Unknown Date')
    raw_text = transcript_data.get('content', '')

    # Build the structured Header
    header = f"SYMBOL: {sym}\n"
    header += f"QUARTER: Q{quarter}\n"
    header += f"YEAR: {year}\n"
    header += f"DATE/TIME: {date_time}\n"
    header += f"--- START OF TRANSCRIPT ---\n\n"

    # Clean and format the text body
    import re
    import textwrap

    # Clean up whitespace first
    clean_text = " ".join(raw_text.split())

    # Find speaker patterns (e.g., "John Smith:" or "Operator:")
    # Insert line breaks before each speaker
    formatted_text = re.sub(
        r'([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\s*:',
        r'\n\n\1:',
        clean_text
    )

    # Make speaker names uppercase (name followed by colon at start of line)
    def uppercase_speaker(match):
        return match.group(1).upper() + ':'

    formatted_text = re.sub(
        r'^([A-Za-z]+(?:\s+[A-Za-z]+)*):',
        uppercase_speaker,
        formatted_text,
        flags=re.MULTILINE
    )

    # For print/file: wrap at 80 characters for readability
    if output in ('print', 'file'):
        paragraphs = formatted_text.strip().split('\n\n')
        wrapped_paragraphs = []
        for para in paragraphs:
            wrapped = textwrap.fill(para.strip(), width=80)
            wrapped_paragraphs.append(wrapped)
        clean_body = '\n\n'.join(wrapped_paragraphs)
    else:
        # For LLM use: no wrapping, just clean paragraphs
        clean_body = formatted_text.strip()

    full_output = header + clean_body

    # Handle output flags
    if output == 'print':
        print(full_output)
        return None

    elif output == 'file':
        filename = f"{sym}_{year}_Q{quarter}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_output)
        print(f"Transcript with header saved to {filename}")
        return None

    return full_output  # Default: return LLM-optimized string


#-------------------------------------------------------

def fmp_empCount(symbol):
    '''
    Historical Employee Count from SEC filings
    input: symbol as string (stock ticker)
    returns: Series with filingDate as index and employeeCount as values
    '''
    url = f"https://financialmodelingprep.com/api/v4/historical/employee_count?symbol={symbol}&apikey={apikey}"
    response = urlopen(url, context=ssl_context)
    data = response.read().decode("utf-8")
    stuff = json.loads(data)

    df = pd.DataFrame(stuff)
    s = df.set_index('filingDate')['employeeCount']
    s.index = pd.to_datetime(s.index)
    s = s.sort_index(ascending=True)
    s.name = 'employeeCount'
    return s

#-------------------------------------------------------------

def fmp_etfExposure(symbol):
    """
    Fetches ETF stock exposure for a given ticker symbol, filters results, 
    removes the 'assetExposure' column, and sets 'etfSymbol' as the index.
    """
    # Uses the global 'apikey' variable defined in your notebook
    url = f"https://financialmodelingprep.com/api/v3/etf-stock-exposure/{symbol}?apikey={apikey}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # 1. Filtering logic: length < 5 and no "."
        filtered_df = df[
    (df['etfSymbol'].str.len() < 5) & 
    (~df['etfSymbol'].str.contains(r'\.', regex=True))
].copy()
        
        # 2. Drop "assetExposure" since it's redundant (always the input symbol)
        # 3. Set "etfSymbol" as the index
        final_df = filtered_df.drop(columns=['assetExposure']).set_index('etfSymbol')
        
        # 4. Sort descending by weightPercentage
        sorted_df = final_df.sort_values(by='weightPercentage', ascending=False)
        
        return sorted_df
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()