import streamlit as st
import urllib.parse
import http.client
import json
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px


# Function to search for symbols
def get_suggestion_tickers(q):
    # Define the parameters
    params = {
        "q": urllib.parse.quote(q),
        "lang": "en-US",
        "region": "In",
        "enableFuzzyQuery": True,
    }

    # Construct the query string
    query_string = '&'.join([f"{key}={params[key]}" for key in params])

    # Establish connection
    conn = http.client.HTTPSConnection("query1.finance.yahoo.com")
    # Send GET request
    conn.request("GET", f"/v1/finance/search?{query_string}")

    # Get response
    res = conn.getresponse()
    data = []
    if res.status == 200:
        data = json.loads(res.read())['quotes']
    else:
        print("Failed to fetch data:", res.status)

    # Close connection
    conn.close()
    return data


def get_daily_prices_in_inr(ticker):
    # Get the data
    data = yf.download(ticker, interval="1d")['Adj Close']
    # Convert to INR
    info = yf.Ticker(ticker).info
    currency = info['currency']
    if currency == 'INR':
        return data
    exchange_rates = yf.download(f'{currency}INR=X', interval="1d")['Adj Close']
    common_dates = data.index.intersection(exchange_rates.index)
    return data[common_dates] * exchange_rates[common_dates]


def get_efficient_frontier(tickers):
    prices = [get_daily_prices_in_inr(t) for t in tickers]

    rolling_returns_252_days = [p.pct_change(252).dropna() for p in prices]
    common_index = rolling_returns_252_days[0].index
    for ret in rolling_returns_252_days[1:]:
        common_index = common_index.intersection(ret.index)

    aligned_returns = [ret.loc[common_index] for ret in rolling_returns_252_days]
    returns_df = pd.concat(aligned_returns, axis=1)

    mean_returns = returns_df.mean()

    cov_matrix = returns_df.cov()
    num_portfolios = 40000
    num_tickers = len(tickers)
    results = np.zeros((3 + num_tickers, num_portfolios))

    for i in range(num_portfolios):
        weights = np.random.random(num_tickers)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights) * 100
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * 100
        results[0, i] = round(portfolio_return, 2)
        results[1, i] = round(portfolio_std_dev, 2)
        results[2, i] = round((results[0, i] - 4) / results[1, i], 2)
        for j in range(len(weights)):
            results[j + 3, i] = round(weights[j], 2)
    return results


def generate_plot(tickers):
    names = []
    for t in st.session_state.portfolio_tickers:
        if 'longname' in t:
            names.append(t['longname'])
        elif 'shortname' in t:
            names.append(t['shortname'])
        else:
            names.append(t['symbol'])
    if len(tickers) > 0:
        results = get_efficient_frontier(tickers)
        max_sharpe_idx = np.argmax(results[2])
        max_sharpe_return = results[0, max_sharpe_idx]
        max_sharpe_std_dev = results[1, max_sharpe_idx]
        max_sharpe_ratio = results[2, max_sharpe_idx]
        min_risk_idx = np.argmin(results[1])
        min_risk_return = results[0, min_risk_idx]
        min_risk_std_dev = results[1, min_risk_idx]
        min_risk_sharpe_ratio = results[2, min_risk_idx]

        columns = ['Return', 'Risk', 'Sharpe Ratio'] + tickers
        results_df = pd.DataFrame(results.T, columns=columns)
        for i in range(len(tickers)):
            results_df[names[i]] = results_df[tickers[i]]
        results_df = results_df.drop(tickers, axis=1)
        # Create Plotly figure
        fig = px.scatter(results_df, x='Risk', y='Return', color='Sharpe Ratio', hover_name=results_df.index,
                         hover_data=names,
                         labels={'Risk': 'Portfolio Risk (Std. Deviation)', 'Return': 'Portfolio Return'},
                         title='Portfolio Optimization',
                         color_continuous_scale=px.colors.sequential.Inferno)
        fig.add_trace(
            px.scatter(results_df.iloc[[max_sharpe_idx]], x='Risk', y='Return', color='Sharpe Ratio').data[0])
        fig.add_trace(
            px.scatter(results_df.iloc[[min_risk_idx]], x='Risk', y='Return', color='Sharpe Ratio').data[0])
        st.plotly_chart(fig)

        # Highlight the max sharpe and min risk portfolios
        st.write(
            f"Max Sharpe Ratio Portfolio: Return = {max_sharpe_return:.2f}%, Risk = {max_sharpe_std_dev:.2f}%, Sharpe Ratio = {max_sharpe_ratio:.2f}")
        st.write(
            f"Min Risk Portfolio: Return = {min_risk_return:.2f}%, Risk = {min_risk_std_dev:.2f}%, Sharpe Ratio = {min_risk_sharpe_ratio:.2f}")


if 'portfolio_tickers' not in st.session_state:
    st.session_state.portfolio_tickers = []


def main():
    st.set_page_config(page_title="Efficient Frontier", page_icon=":chart_with_upwards_trend:")
    st.title("Find Efficient frontier for your portfolio")

    # Input field for autocomplete
    input_text = st.text_input("Search security:")

    # Get suggestions based on input text
    suggestions = get_suggestion_tickers(input_text)

    # Show autocomplete options
    displays = []
    for suggestion in suggestions:
        if 'longname' in suggestion:
            displays.append(f"({suggestion['symbol']}) - {suggestion['longname']}")
        elif 'shortname' in suggestion:
            displays.append(f"({suggestion['symbol']}) - {suggestion['shortname']}")
        else:
            displays.append(f"({suggestion['symbol']})")
    selected_option = st.selectbox("Choose one security:", displays, index=0)
    if st.button("Add to portfolio"):
        if selected_option:
            selected_ticker = suggestions[displays.index(selected_option)]
            if selected_ticker not in st.session_state.portfolio_tickers:
                st.session_state.portfolio_tickers.append(selected_ticker)

    for i in range(len(st.session_state.portfolio_tickers)):
        item = f"{i+1}. "
        if 'longname' in st.session_state.portfolio_tickers[i]:
            item += st.session_state.portfolio_tickers[i]['longname'] + " - "
        elif 'shortname' in st.session_state.portfolio_tickers[i]:
            item += st.session_state.portfolio_tickers[i]['shortname'] + " - "
        item += st.session_state.portfolio_tickers[i]['symbol']
        st.write(item)
        # delete button
        if st.button(f"Delete {st.session_state.portfolio_tickers[i]['symbol']}"):
            st.session_state.portfolio_tickers.pop(i)
            st.rerun()

    if st.button("Clear portfolio"):
        st.session_state.portfolio_tickers = []
        st.rerun()

    if st.button("Show Efficient Frontier"):
        tickers = [t['symbol'] for t in st.session_state.portfolio_tickers]
        if len(tickers) > 0:
            generate_plot(tickers)
        else:
            st.write("Please add some tickers to the portfolio")


if __name__ == "__main__":
    main()
