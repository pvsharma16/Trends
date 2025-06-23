import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

CSV_PATH = "fno_stocks.csv"
WINDOW_SIZE = 7
START_DATE = "2021-06-01"
END_DATE = "2024-06-01"

# --- Load metadata
@st.cache_data
def load_stock_metadata():
    df = pd.read_csv(CSV_PATH)
    return df.dropna(subset=['symbol'])

# --- Fetch price data
@st.cache_data
def fetch_data(tickers):
    data = yf.download(tickers, start=START_DATE, end=END_DATE)
    adj = data['Adj Close'] if 'Adj Close' in data else data
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    return adj.dropna()

# --- Compute trend vectors (no cache!)
def compute_trends(df):
    trends = []
    for ticker in df.columns:
        series = df[ticker].dropna()
        for i in range(len(series) - WINDOW_SIZE + 1):
            win = series.iloc[i:i+WINDOW_SIZE].values.reshape(-1, 1)
            z = StandardScaler().fit_transform(win).flatten().tolist()
            trends.append({
                "ticker": ticker,
                "start_date": series.index[i],
                "vector": z
            })
    return pd.DataFrame(trends)

# --- PCA + clustering (no cache!)
def cluster_trends(trend_df, n_clusters=5):
    X = np.stack(trend_df['vector'].values)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    trend_df["pca_1"] = X_pca[:, 0]
    trend_df["pca_2"] = X_pca[:, 1]
    km = KMeans(n_clusters=n_clusters, random_state=42)
    trend_df["cluster"] = km.fit_predict(X_pca)
    return trend_df

# --- Main
def main():
    st.title("🧪 Minimal Trend Cluster Debug App")

    stock_meta = load_stock_metadata()
    sectors = sorted(stock_meta['sector'].unique())
    caps = sorted(stock_meta['market_cap'].unique())

    selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)
    selected_caps = st.sidebar.multiselect("Market Cap", caps, default=caps)

    filtered = stock_meta[
        stock_meta['sector'].isin(selected_sectors) &
        stock_meta['market_cap'].isin(selected_caps)
    ]
    tickers = filtered['symbol'].tolist()

    if len(tickers) == 0:
        st.warning("No tickers selected.")
        return

    st.write(f"⏳ Loading data for {len(tickers)} tickers...")
    prices = fetch_data(tickers)
    st.success("✅ Data loaded.")

    st.write("⚙️ Computing trends...")
    try:
        trends = compute_trends(prices)
        st.success(f"✅ Computed {len(trends)} trend windows.")
    except Exception as e:
        st.error(f"❌ Error in compute_trends: {e}")
        return

    st.write("🔬 Clustering trends...")
    try:
        trends = cluster_trends(trends)
    except Exception as e:
        st.error(f"❌ Error in clustering: {e}")
        return

    trends = trends.merge(stock_meta, left_on='ticker', right_on='symbol', how='left')
    fig = px.scatter(trends, x='pca_1', y='pca_2', color='sector', hover_data=['ticker', 'market_cap'])
    st.plotly_chart(fig, use_container_width=True)

    cluster = st.sidebar.selectbox("Cluster", sorted(trends['cluster'].unique()))
    sample = trends[trends['cluster'] == cluster].head(5)
    for _, row in sample.iterrows():
        st.caption(f"{row['ticker']} | {row['start_date'].date()} | {row['sector']} | {row['market_cap']}")
        st.line_chart(row['vector'])

if __name__ == "__main__":
    main()
