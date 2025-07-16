import re 
import os
import pickle
import hashlib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
plt.style.use('dark_background')

from pykrx import stock
from fredapi import Fred
from threading import RLock
from datetime import datetime, timedelta

from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="sk-or-v1-CENSORED-KEY",
)

MODEL="google/gemini-2.5-flash"

DATA = "data"
FORECAST = "forecast"
CACHE = {
    DATA: {},
    FORECAST: {}
}

def load_cache():
    if os.path.exists('cache.pkl'):
        with open('cache.pkl', 'rb') as f:
            return pickle.load(f)
    return CACHE

def save_cache():
    with open('cache.pkl', 'wb') as f:
        pickle.dump(CACHE, f)

# Load cache if exists
CACHE = load_cache()

HISTORY_DAYS = 20
FORECAST_DAYS = 5

ticker_data = pd.read_csv("ticker_data.csv", encoding='utf-8')


def get_ticker_code_from_name(ticker_name: str) -> str | None:
    row = ticker_data[ticker_data['ticker_name'] == ticker_name]
    if row.empty:
        return None
    return row.iloc[0]['ticker_code']

def split_date_range(start_date: str, end_date: str) -> list:
    start = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")
    delta = end - start
    split_ranges = []
    # Split the date range into 2-year intervals
    for i in range(0, delta.days + 1, 365 * 2):
        split_start = start + timedelta(days=i)
        split_end = min(end, split_start + timedelta(days=365 * 2 - 1))
        split_ranges.append((split_start.strftime("%Y%m%d"), split_end.strftime("%Y%m%d")))
    return split_ranges

def get_fred_data(series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    key = f"{series_id}_{start_date}_{end_date}"
    hashed_cache_name = hashlib.sha1(key.encode()).hexdigest()
    cache_file = f"data/{hashed_cache_name}.csv"
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        df_cached = pd.read_csv(cache_file)
        return df_cached
    except FileNotFoundError:
        fred = Fred(api_key='1813ca70b0692eac480e11a0691dac96')
        df = fred.get_series(series_id, start_date, end_date)
        df = df.reset_index()
        df.columns = ['date', series_id]
        df.to_csv(cache_file, index=False)
        return df

def get_stock_data(ticker_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    key = f"{ticker_code}_{start_date}_{end_date}"
    hashed_cache_name = hashlib.sha1(key.encode()).hexdigest()
    cache_file = f"data/{hashed_cache_name}.csv"
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        if not os.path.exists(cache_file):
            raise FileNotFoundError()
        print(f"Loading cached data for {ticker_code} from {cache_file}")
        try:
            df_cached = pd.read_csv(cache_file)
        except BaseException:
            os.remove(cache_file)
            raise FileNotFoundError()
        return df_cached
    except FileNotFoundError:
        df_ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, ticker_code)
        df_ohlcv.reset_index(inplace=True)
        df_ohlcv.rename(columns={'날짜':'date', '시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume'}, inplace=True)
        df_ohlcv = df_ohlcv[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        df_fundamental = stock.get_market_fundamental_by_date(start_date, end_date, ticker_code)
        df_fundamental.reset_index(inplace=True)
        df_fundamental.rename(columns={'날짜':'date'}, inplace=True)
        df_ohlcv = pd.merge(df_ohlcv, df_fundamental, on='date', how='left')

        df_trading_value = stock.get_market_trading_value_by_date(start_date, end_date, ticker_code)
        df_trading_value.reset_index(inplace=True)
        df_trading_value.rename(columns={
            '날짜': 'date',
            '기관합계': 'institution',
            '외국인합계': 'foreign',
            '개인': 'individual',
            '기타법인': 'other_corporation',
        }, inplace=True)
        df_trading_value.drop(columns=['전체'], inplace=True)
        df_ohlcv = pd.merge(df_ohlcv, df_trading_value, on='date', how='left')

        df_trading_volume = stock.get_market_trading_volume_by_date(start_date, end_date, ticker_code)
        df_trading_volume.reset_index(inplace=True)
        df_trading_volume.rename(columns={
            '날짜': 'date',
            '기관합계': 'institution_volume',
            '외국인합계': 'foreign_volume',
            '개인': 'individual_volume',
            '기타법인': 'other_corporation_volume',
        }, inplace=True)
        df_trading_volume.drop(columns=['전체'], inplace=True)
        df_ohlcv = pd.merge(df_ohlcv, df_trading_volume, on='date', how='left')
        
        split_ranges = split_date_range(start_date, end_date)
        df_shortings = []
        for start, end in split_ranges:
            df_shorting = stock.get_shorting_volume_by_date(start, end, ticker_code)
            df_shorting.reset_index(inplace=True)
            df_shortings.append(df_shorting)
        df_shorting_volume = pd.concat(df_shortings, ignore_index=True)
        df_shorting_volume.rename(columns={
            '날짜': 'date',
            '공매도': 'shorting_volume',
            '매수': 'buy_volume',
            '비중': 'shorting_ratio',
        }, inplace=True)
        df_ohlcv = pd.merge(df_ohlcv, df_shorting_volume, on='date', how='left')

        df_kospi = get_index_data("1001", start_date, end_date)
        df_kospi.rename(columns={'close':'kospi_close'}, inplace=True)
        df_ohlcv['date'] = df_ohlcv['date'].astype(str)
        df_kospi['date'] = df_kospi['date'].astype(str)
        df_stock = pd.merge(df_ohlcv, df_kospi[['date', 'kospi_close']], on='date', how='left')

        # Define FRED series IDs to fetch
        fred_series = {
            'DEXKOUS': 'exchange_rate', # KRW/USD Exchange Rate
            'DGS10': 'us_10y_yield'     # 10-Year Treasury Constant Maturity Rate
        }

        df_stock['date'] = pd.to_datetime(df_stock['date'])

        for series_id, col_name in fred_series.items():
            df_fred = get_fred_data(series_id, start_date, end_date)
            df_fred.rename(columns={series_id: col_name}, inplace=True)
            df_fred['date'] = pd.to_datetime(df_fred['date'])
            df_stock = pd.merge(df_stock, df_fred, on='date', how='left')


        # Forward-fill missing values (for weekends/holidays) and then back-fill
        df_stock.ffill(inplace=True)
        df_stock.bfill(inplace=True)

        # Make a clean copy to add features to
        df_with_ta = df_stock.copy()

        # 1. Moving Average Convergence Divergence (MACD)
        indicator_macd = MACD(close=df_with_ta['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
        df_with_ta['macd'] = indicator_macd.macd()
        df_with_ta['macd_signal'] = indicator_macd.macd_signal()

        # 2. Relative Strength Index (RSI)
        df_with_ta['rsi'] = RSIIndicator(close=df_with_ta['close'], window=14, fillna=True).rsi()

        # 3. Bollinger Bands (BB)
        indicator_bb = BollingerBands(close=df_with_ta['close'], window=20, window_dev=2, fillna=True)
        df_with_ta['bb_hband'] = indicator_bb.bollinger_hband() # Upper band
        df_with_ta['bb_lband'] = indicator_bb.bollinger_lband() # Lower band
        df_with_ta['bb_pband'] = indicator_bb.bollinger_pband() # Percentage band
        df_with_ta['bb_wband'] = indicator_bb.bollinger_wband() # Width band

        # 4. On-Balance Volume (OBV)
        df_with_ta['obv'] = OnBalanceVolumeIndicator(close=df_with_ta['close'], volume=df_with_ta['volume'], fillna=True).on_balance_volume()

        # The original df_stock is now the one with the new, selected features
        df_stock = df_with_ta

        # Drop first row
        df_stock.drop(index=df_stock.index[0], inplace=True)

        df_stock.to_csv(cache_file, index=False)
        print(f"Fetched new data and saved to cache for {ticker_code}")
        return df_stock
    
def get_index_data(index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    key = f"{index_code}_{start_date}_{end_date}"
    hashed_cache_name = hashlib.sha1(key.encode()).hexdigest()
    cache_file = f"data/{hashed_cache_name}.csv"
    if not os.path.exists('data'):
        os.makedirs('data')
    try:
        df_cached = pd.read_csv(cache_file)
        return df_cached
    except FileNotFoundError:
        df_ohlcv = stock.get_index_ohlcv_by_date(start_date, end_date, index_code)
        df_ohlcv.reset_index(inplace=True)
        df_ohlcv.rename(columns={'날짜':'date', '시가':'open', '고가':'high', '저가':'low', '종가':'close', '거래량':'volume'}, inplace=True)
        df_ohlcv = df_ohlcv[['date', 'open', 'high', 'low', 'close', 'volume']]
        df_ohlcv.to_csv(cache_file, index=False)
        return df_ohlcv
    
def get_latest_stock_data(ticker_code: str) -> pd.DataFrame:
    today = datetime.now().strftime("%Y%m%d")
    cache_id = f"{ticker_code},{today}"
    if cache_id in CACHE[DATA]:
        return CACHE[DATA][cache_id]
    data = get_stock_data(ticker_code, "20250101", today)[-HISTORY_DAYS:]
    CACHE[DATA][cache_id] = data
    save_cache()
    return data

def get_prompt_for_forecast(ticker_code: str) -> str:
    datas_str = ""
    history_data = get_latest_stock_data(ticker_code)
    datas = history_data.to_dict(orient='records')
    for data in datas:
        datas_str += str(data["date"]).split(" ")[0] + "\t"
        for i, (k, v) in enumerate(data.items()):
            # v = round(v, 2) if isinstance(v, (int, float)) else str(v)
            data[k] = round(v, 2) if isinstance(v, (int, float)) else str(v)
        datas_str += "\t".join([f"{k}: {str(v).replace('\n', '')}" for k, v in data.items() if k != "date"]) + "\n"
    datas_str = datas_str.strip()

    prompt_str = f"""Here is the Stock data of {ticker_code} for the past {HISTORY_DAYS} days:
I will now give you data for the past {HISTORY_DAYS} recorded dates, and please help me forecast the data for next {FORECAST_DAYS} recorded dates. The data is as follows:
```
{datas_str}
```
Please give me the close data for the next {FORECAST_DAYS} recorded dates, remember to give me the close data. 
You must first conduct reasoning inside <think> …</think>. 
When you have the final answer, you can output the answer inside <answer>…</answer> and the reason of the answer inside <reason>…</reason>.

Example output:
```
<think>...</think>
<answer>20XX-XX-XX\tclose: XXXXX
20XX-XX-XX\tclose: XXXXX
... (continue for {FORECAST_DAYS} days)</answer>
<reason>...</reason>
```
"""
    return prompt_str

def get_forecast(prompt, model):
    if prompt in CACHE[FORECAST]:
        return CACHE[FORECAST][prompt]
    
    completion = client.chat.completions.create(
      model=model,
      messages=[
        {
          "role": "user",
          "content": prompt
        }
      ],
      reasoning_effort="high",
    )
    result = str(completion.choices[0].message.content)
    # Extract the <reason>...</reason> content
    reason_match = re.search(r'<reason>(.*?)</reason>', result, re.DOTALL)
    if not reason_match:
        raise ValueError(f"No valid <reason> section found in the response: {result}")
    reason_content = reason_match.group(1).strip()
    
    # Extract <answer>...</answer> content
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, result, re.DOTALL)
    if not answer_match:
        raise ValueError(f"No valid <answer> section found in the response: {result}")
    
    answer = answer_match.group(1).strip()
    lines = answer.split('\n')
    forecast = {}
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            date = parts[0].strip()
            close_value = parts[1].split(':')[1].strip().replace(',', '')
            forecast[date] = float(close_value)

    CACHE[FORECAST][prompt] = (reason_content, forecast)
    save_cache()
    return reason_content, forecast

_lock = RLock()

st.title("Stock Forecasting with LLM")
st.write(f"과거 {HISTORY_DAYS}일의 주식 데이터를 기반으로 다음 {FORECAST_DAYS}일의 종가를 예측합니다.")

# Streamlit UI for user input
ticker_name = st.text_input("주가 종목 이름/코드를 입력하세요: ", "삼성전자")
if ticker_name:
    ticker_code = get_ticker_code_from_name(ticker_name)
    if ticker_code is None:
        st.error(f"종목 이름 '{ticker_name}'에 해당하는 종목 코드를 찾을 수 없습니다.")
        st.stop()
    st.write(f"종목 코드: {ticker_code}")

    password = st.text_input("비밀번호를 입력하세요: ", type="password")
    if password != "CENSORED-PASSWORD":
        st.error("비밀번호가 올바르지 않습니다.")
        st.stop()
    
    try:
        if st.button("예측 시작"):
            with st.spinner("데이터 가져오는 중...", show_time=True):
                history_data = get_latest_stock_data(ticker_code)
            with st.spinner("데이터 정리 중...", show_time=True):
                prompt = get_prompt_for_forecast(ticker_code)
            with st.spinner("예측 중...", show_time=True):
                print(f"=== Forecasting for {ticker_name} ({ticker_code}) ===")
                reason, forecast = get_forecast(prompt, MODEL)
                st.write("예측 결과:")
                # Display history and forecast data with matplotlib
                with _lock:
                    fig, ax = plt.subplots()
                    history_data['date'] = pd.to_datetime(history_data['date'])
                    ax.plot(history_data['date'], history_data['close'], label='Historical Close', color='blue')
                    # 과거 마지막 날과 예측 첫날을 이어주기 위해 과거 마지막 종가 추가
                    forecast_dates = [history_data['date'].iloc[-1]] + list(forecast.keys())
                    forecast_values = [history_data['close'].iloc[-1]] + list(forecast.values())
                    forecast_dates = pd.to_datetime(forecast_dates)
                    ax.plot(forecast_dates, forecast_values, label='Forecast Close', color='orange', linestyle='--')
                    # ax.plot(forecast_dates, forecast_values, label='Forecast Close', color='orange', linestyle='--')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Close Price')
                    ax.set_title(f"{ticker_name} ({ticker_code}) - Stock Price Forecast")
                    ax.grid(color='gray', linestyle='--', linewidth=0.5)
                    fig.autofmt_xdate()
                    ax.legend()
                    st.pyplot(fig)

                # Display the reason and forecast
                st.write("예측 이유:")
                st.write(reason)
                
                print(f"=== Forecasted for {ticker_name} ({ticker_code}) ===")
    except Exception as e:
        st.error(f"오류 발생: {e}")
