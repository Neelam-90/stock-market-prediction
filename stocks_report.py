import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from stocknews import StockNews

st.title("STOCK MARKET ANALYSIS")

# Predefined list of common company names and their tickers
common_companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Facebook": "META",
    "Netflix": "NFLX",
    "Nvidia": "NVDA",
    "Alibaba": "BABA",
    "JPMorgan": "JPM",
    "Berkshire Hathaway": "BRK-B",
    "Johnson & Johnson": "JNJ",
    "Visa": "V",
    "Walmart": "WMT",
    "Samsung": "005930.KS",
    "Nestle": "NSRGY",
    "Procter & Gamble": "PG",
    "UnitedHealth Group": "UNH",
    "Mastercard": "MA",
    "Kweichow Moutai": "600519.SS",
    "Roche": "RHHBY",
    "LVMH": "LVMUY",
    "Intel": "INTC",
    "Coca-Cola": "KO",
    "Pfizer": "PFE",
    "Nike": "NKE",
    "McDonald's": "MCD",
    "PepsiCo": "PEP",
    "Cisco Systems": "CSCO",
    "Chevron": "CVX",
    "Adobe": "ADBE",
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Larsen & Toubro": "LT.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Wipro": "WIPRO.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Axis Bank": "AXISBANK.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Ultratech Cement": "ULTRACEMCO.NS",
    "Nestle India": "NESTLEIND.NS",
    "Dr. Reddy's Laboratories": "DRREDDY.NS",
    "HDFC Life Insurance": "HDFCLIFE.NS",
    "Hindalco": "HINDALCO.NS",
    "Titan Company": "TITAN.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bharat Petroleum": "BPCL.NS",
    "Indian Oil": "IOC.NS",
    "Power Grid Corporation": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Vedanta": "VEDL.NS",
    "Tech Mahindra": "TECHM.NS",
    "Grasim Industries": "GRASIM.NS",
    "Britannia Industries": "BRITANNIA.NS",
    "Shree Cement": "SHREECEM.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Havells India": "HAVELLS.NS",
    "Pidilite Industries": "PIDILITIND.NS",
    "Dabur India": "DABUR.NS",
    "Godrej Consumer Products": "GODREJCP.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Others": "Others"
    
}

company_names = list(common_companies.keys())

# Sidebar inputs
option = st.sidebar.selectbox("Select a Company", company_names)

if option == "Others":
    st.sidebar.write("No other company available, please enter valid stocks.")
    company_name = None
    ticker = None
else:
    company_name = option
    ticker = common_companies[company_name]

start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

# Check if the ticker and dates are provided
if ticker and start_date and end_date:
    # Fetch data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    if not data.empty:
        # Display data
        st.subheader("Stock Data")
        st.write(data)

        # Plot adjusted close and open prices on the same graph
        fig = px.line(data, x=data.index, y=['Adj Close', 'Open'], title=f'{ticker} Prices')
        st.plotly_chart(fig)

        # Tabs for different analyses
        pricing_data, fundamental_data, news, prediction = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News", "Prediction"])

        with pricing_data:
            st.header('Price Movements')
            data2 = data.copy()
            data2['% Change'] = data['Adj Close'].pct_change()
            data2.dropna(inplace=True)
            st.write(data2)
            
            annual_return = data2['% Change'].mean() * 252 * 100
            st.write("Annual Return is", annual_return, '%')
            
            stdev = np.std(data2['% Change']) * np.sqrt(252)
            st.write('Standard Deviation is', stdev * 100, '%')
            
            risk_adj_return = annual_return / (stdev * 100)
            st.write('Risk Adj. Return is', risk_adj_return)

        with fundamental_data:
            st.header('Fundamental Data')
            stock = yf.Ticker(ticker)
            
            st.subheader('Balance Sheet')
            balance_sheet = stock.balance_sheet
            st.write(balance_sheet)
            
            st.subheader('Income Statement')
            income_statement = stock.financials
            st.write(income_statement)
            
            st.subheader('Cash Flow Statement')
            cash_flow = stock.cashflow
            st.write(cash_flow)

        with news:
            st.header(f'News for {ticker}')
            sn = StockNews(ticker, save_news=False)
            df_news = sn.read_rss()
            
            for i in range(10):
                st.subheader(f'News {i+1}')
                st.write(df_news['published'][i])
                st.write(df_news['title'][i])
                st.write(df_news['summary'][i])
                
                title_sentiment = df_news['sentiment_title'][i]
                st.write(f'Title Sentiment: {title_sentiment}')
                
                news_sentiment = df_news['sentiment_summary'][i]
                st.write(f'News Sentiment: {news_sentiment}')

        with prediction:
            st.header('Stock Price Prediction')
            
            # Prepare data for prediction
            data['Date'] = data.index
            data['Days'] = (data['Date'] - data['Date'].min()).dt.days
            X = data[['Days']]
            y = data['Adj Close']
            
            # Split data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train a linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict the future stock prices
            future_days = st.slider('Days to Predict into the Future', 1, 30, 5)
            future_X = pd.DataFrame({'Days': np.arange(X['Days'].max() + 1, X['Days'].max() + 1 + future_days)})
            future_predictions = model.predict(future_X)
            
            # Plot the predictions
            future_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=future_days)
            prediction_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
            
            fig = px.line(prediction_df, x='Date', y='Predicted Price', title='Future Stock Price Prediction')
            st.plotly_chart(fig)
            
            # Display predictions
            st.subheader('Future Predictions')
            st.write(prediction_df)
    else:
        st.write("No data found for the specified ticker and date range.")
else:
    st.write("Please select a valid company name or ticker and date range.")




