import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Add custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    """
    <style>
    h1 {
        color: #333333;
        text-align: center;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .stMarkdown {
        color: #555555;
        line-height: 1.6;
    }
    .stDataFrame {
        border: 1px solid #dddddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stock Market Analysis and Forecasting App")

# User Input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL")

# Fetch Stock Data
if ticker:
    st.subheader("Stock Data")
    stock_data = yf.download(ticker, start="2015-01-01")
    
    # Flatten MultiIndex columns if present
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = [' '.join(col).strip() for col in stock_data.columns]
    
    # Ensure the data is not empty
    if stock_data.empty:
        st.error("No data was returned for the given stock ticker. Please check the ticker symbol or try again later.")
        st.stop()
    
    # Handle missing 'Close' column
    if 'Close' not in stock_data.columns:
        st.write("Debugging: Available columns in stock_data:", stock_data.columns.tolist())
        possible_close_columns = [col for col in stock_data.columns if 'Close' in col]
        if possible_close_columns:
            st.warning(f"'Close' column is missing. Using '{possible_close_columns[0]}' as a fallback.")
            stock_data['Close'] = stock_data[possible_close_columns[0]]
        else:
            st.error("The 'Close' column is missing and cannot be reconstructed. Please check the data source or try a different stock ticker.")
            st.stop()
    
    # Display the structure of the downloaded data
    st.write("Downloaded data structure:")
    st.write(stock_data.head())
    
    # Visualizations
    st.write(stock_data.tail())
    st.line_chart(stock_data['Close'])
    
    st.subheader("Volume Traded Over Time")
    if 'Volume' not in stock_data.columns:
        st.warning("'Volume' column is missing. Skipping volume chart.")
    else:
        st.bar_chart(stock_data['Volume'])
    
    st.subheader("Daily Returns")
    daily_returns = stock_data['Close'].pct_change().dropna()
    st.line_chart(daily_returns)
    
    st.subheader("Correlation Matrix")
    correlation_matrix = stock_data.corr()
    st.write(correlation_matrix)
    fig, ax = plt.subplots()
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    st.pyplot(fig)
    
    # Additional Graphics
    st.subheader("Sector Distribution (Example Pie Chart)")
    sector_data = {'Technology': 40, 'Healthcare': 25, 'Finance': 20, 'Energy': 15}
    fig, ax = plt.subplots()
    ax.pie(sector_data.values(), labels=sector_data.keys(), autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)
    
    st.subheader("Closing Price Distribution (Histogram)")
    fig, ax = plt.subplots()
    ax.hist(stock_data['Close'], bins=20, color='blue', alpha=0.7)
    ax.set_title("Distribution of Closing Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    # Moving Averages
    st.subheader("Moving Averages")
    # Ensure 'Close' column exists
    if 'Close' not in stock_data.columns:
        if 'Adj Close' in stock_data.columns:
            stock_data['Close'] = stock_data['Adj Close']
        else:
            st.error("The 'Close' column is missing and cannot be reconstructed.")
            st.stop()
    
    # Calculate moving averages
    stock_data['20-day MA'] = stock_data['Close'].rolling(window=20).mean()
    stock_data['50-day MA'] = stock_data['Close'].rolling(window=50).mean()
    
    # Verify moving averages are calculated
    if stock_data[['20-day MA', '50-day MA']].isnull().all().all():
        st.error("Moving averages could not be calculated. Please check the data.")
        st.stop()
    # Ensure required columns exist before plotting
    if all(col in stock_data.columns for col in ['Close', '20-day MA', '50-day MA']):
        st.line_chart(stock_data[['Close', '20-day MA', '50-day MA']])
    else:
        st.error("Required columns for moving averages are missing. Please check the data processing steps.")
    
    # Bollinger Bands
    st.subheader("Bollinger Bands")
    stock_data['Upper Band'] = stock_data['20-day MA'] + 2 * stock_data['Close'].rolling(window=20).std()
    stock_data['Lower Band'] = stock_data['20-day MA'] - 2 * stock_data['Close'].rolling(window=20).std()
    st.line_chart(stock_data[['Close', 'Upper Band', 'Lower Band']])
    
    # Candlestick Chart
    st.subheader("Candlestick Chart")
    import plotly.graph_objects as go
    if 'Open' not in stock_data.columns:
        st.warning("'Open' column is missing. Skipping candlestick chart.")
    else:
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close']
                )
            ]
        )
        st.plotly_chart(fig)
    
    # RSI (Relative Strength Index)
    st.subheader("RSI (Relative Strength Index)")
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    stock_data['RSI'] = rsi
    st.line_chart(stock_data['RSI'])
    
    # MACD (Moving Average Convergence Divergence)
    st.subheader("MACD (Moving Average Convergence Divergence)")
    short_ema = stock_data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = stock_data['Close'].ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=9, adjust=False).mean()
    stock_data['MACD'] = macd
    stock_data['Signal Line'] = signal
    st.line_chart(stock_data[['MACD', 'Signal Line']])
    
    # Stock Price Prediction
    st.subheader("Stock Price Prediction")
    st.write("Using a simple linear regression model for demonstration.")
    from sklearn.linear_model import LinearRegression
    import numpy as np
    days = np.arange(len(stock_data)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, stock_data['Close'])
    future_days = np.arange(len(stock_data) + 365).reshape(-1, 1)
    predicted_prices = model.predict(future_days)
    st.line_chart(predicted_prices)
    
    # Additional Statistics
    st.subheader("Stock Statistics")
    mean_price = float(stock_data['Close'].mean())
    median_price = float(stock_data['Close'].median())
    std_dev_price = float(stock_data['Close'].std())
    st.write(f"Mean Closing Price: {mean_price:.2f}")
    st.write(f"Median Closing Price: {median_price:.2f}")
    st.write(f"Standard Deviation of Closing Price: {std_dev_price:.2f}")
    
    # Advanced Statistics
    st.subheader("Advanced Stock Statistics")
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    ticker = yf.Ticker(stock_symbol)
    pe_ratio = ticker.info.get('trailingPE', 'N/A')
    eps = ticker.info.get('trailingEps', 'N/A')
    st.write(f"Price-to-Earnings Ratio (P/E): {pe_ratio}")
    st.write(f"Earnings Per Share (EPS): {eps}")
    
    # Forecasting with Prophet
    st.subheader("Forecasting Stock Prices")
    df = stock_data.reset_index()[['Date', 'Close']]
    df.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y' columns
    
    # Train Prophet Model
    model = Prophet()
    model.fit(df)
    
    # Create Future Dataframe
    future = model.make_future_dataframe(periods=365 * 2)  # Forecast for 2 years
    forecast = model.predict(future)
    
    # Plot Forecast
    st.write("Forecasted Stock Prices")
    fig = model.plot(forecast)
    st.pyplot(fig)
    
    # Predict Stock Price at End of 2025
    end_2025 = forecast[forecast['ds'] == '2025-12-31']
    if not end_2025.empty:
        predicted_price = end_2025['yhat'].values[0]
        st.write(f"Predicted Stock Price on 2025-12-31: ${predicted_price:.2f}")