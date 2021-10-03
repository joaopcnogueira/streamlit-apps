import yfinance as yf
import streamlit as st
from pandas_datareader import data as web

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and **volume**.
""")

ticker_symbol = st.text_input("Type in the stock symbol", 'GOOGL')

try:
    ticker_df = web.DataReader(ticker_symbol, data_source='yahoo', start='2015-01-01', end='2021-09-24')
except:
    ticker_symbol = '.'.join([ticker_symbol, 'sa'])
    ticker_df = web.DataReader(ticker_symbol, data_source='yahoo', start='2015-01-01', end='2021-09-24')


st.write("## Closing Price")
st.line_chart(ticker_df['Close'])

st.write("## Volume")
st.line_chart(ticker_df['Volume'])
