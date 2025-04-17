# Stock Market Analysis and Forecasting App

## Overview
This application provides a comprehensive platform for analyzing and forecasting stock market data. It includes various visualizations, statistical analyses, and predictive models to help users make informed decisions.

## Features
- **Stock Data Visualization**: Line charts, bar charts, candlestick charts, and more.
- **Moving Averages**: 20-day and 50-day moving averages for trend analysis.
- **Bollinger Bands**: Visualize price volatility.
- **RSI (Relative Strength Index)**: Measure stock momentum.
- **MACD (Moving Average Convergence Divergence)**: Identify trend changes.
- **Correlation Matrix**: Analyze relationships between stock attributes.
- **Stock Price Prediction**: Forecast future prices using machine learning.
- **Additional Graphics**: Pie chart for sector distribution and histogram for closing price distribution.

## How to Use
1. **Install Dependencies**:
   Ensure you have Python installed. Install the required libraries using:
   ```
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   Start the Streamlit app by running:
   ```
   streamlit run app.py
   ```

3. **Enter Stock Ticker**:
   Input a valid stock ticker symbol (e.g., AAPL, MSFT) to fetch and analyze data.

## Requirements
- Python 3.9 or higher
- Libraries:
  - `streamlit`
  - `yfinance`
  - `prophet`
  - `matplotlib`
  - `pandas`
  - `plotly`
  - `scikit-learn`
  - `numpy`

## File Structure
- `app.py`: Main application file.
- `README.md`: Documentation for the app.
- `requirements.txt`: List of dependencies.

## Future Enhancements
- Add support for more advanced forecasting models.
- Include real-time stock data updates.
- Enhance UI/UX with additional interactive features.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [Yahoo Finance](https://finance.yahoo.com/) for stock data.
- [Streamlit](https://streamlit.io/) for the interactive web app framework.