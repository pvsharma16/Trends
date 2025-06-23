import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import pyarrow as pa
import os

st.set_page_config(layout="wide")

# --- Settings ---
TICKERS = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'ICICIBANK.NS', 'HDFCBANK.NS']
START_DATE = '2021-06-01'
END_DATE = '2024-06-01'
WINDOW_SIZE = 7
PARQUET_PATH = 'data/trend_vectors.parquet'
N_CLUSTERS = 5

# --- Data Loading ---
@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    
    # Always work with 'Adj Close'
    if 'Adj Close' in data.columns:
        adj_close = data['Adj Close']
    else:
        adj_close = data  # Single ticker

    # If user selected just one ticker, Yahoo returns a Series — convert to DataFrame
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    return adj_close.dropna()

@st.cache_data
def compute_trend_vectors(data, window=7):
    trend_vectors = []
    for ticker in data.columns:
        prices = data[ticker].dropna()
        for i in range(0, len(prices) - window + 1):
            window_prices = prices.iloc[i:i + window].values.reshape(-1, 1)
            scaled = StandardScaler().fit_transform(window_prices).flatten()
            trend_vectors.append({
                'ticker': ticker,
                'start_date': prices.index[i],
                'vector': scaled
            })
    return pd.DataFrame(trend_vectors)

def save_to_parquet(df, path):
    df['vector'] = df['vector'].apply(lambda x: np.array(x))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)

def load_from_parquet(path):
    return pq.read_table(path).to_pandas()

# --- PCA + Clustering ---
def reduce_and_cluster(df, n_clusters=5):
    X = np.stack(df['vector'].values)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['pca_1'], df['pca_2'] = X_pca[:, 0], X_pca[:, 1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)
    return df

# --- Main UI ---
def main():
    st.title("📊 Stock Trend Cluster Explorer (7-Day Windows)")
    
    if not os.path.exists(PARQUET_PATH):
        st.info("Fetching and processing data... this will take ~1–2 minutes")
        data = fetch_data(TICKERS, START_DATE, END_DATE)
        trend_df = compute_trend_vectors(data, WINDOW_SIZE)
        save_to_parquet(trend_df, PARQUET_PATH)
        st.success("Trend vectors saved for future use.")
    else:
        trend_df = load_from_parquet(PARQUET_PATH)

    # Reduce + cluster
    trend_df = reduce_and_cluster(trend_df, N_CLUSTERS)

    # UI
    st.sidebar.title("🔍 Cluster Explorer")
    cluster_id = st.sidebar.selectbox("Select Cluster", sorted(trend_df['cluster'].unique()))
    num_samples = st.sidebar.slider("Number of trend previews", 1, 10, 5)

    cluster_df = trend_df[trend_df['cluster'] == cluster_id]
    st.markdown(f"### Cluster {cluster_id} — {len(cluster_df)} trend windows")
    
    st.scatter_chart(cluster_df[['pca_1', 'pca_2']])

    st.subheader("Sample Trend Windows")
    for i in range(min(num_samples, len(cluster_df))):
        sample = cluster_df.iloc[i]
        st.caption(f"{sample['ticker']} | {sample['start_date'].date()}")
        st.line_chart(sample['vector'])

if __name__ == "__main__":
    main()
