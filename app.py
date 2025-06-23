import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
import pyarrow as pa

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(layout="wide")

# --- Config ---
CSV_PATH = 'fno_stocks.csv'
PARQUET_PATH = 'data/trend_vectors.parquet'
START_DATE = '2021-06-01'
END_DATE = '2024-06-01'
WINDOW_SIZE = 7
N_CLUSTERS = 5

# --- Load Stock Metadata ---
@st.cache_data
def load_stock_metadata(path=CSV_PATH):
    df = pd.read_csv(path)
    df.dropna(subset=['symbol'], inplace=True)
    return df

# --- Sidebar Filters ---
def apply_filters(stock_meta):
    st.sidebar.markdown("### Filter Stocks")
    sectors = sorted(stock_meta['sector'].dropna().unique())
    caps = sorted(stock_meta['market_cap'].dropna().unique())

    selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)
    selected_caps = st.sidebar.multiselect("Market Cap", caps, default=caps)

    filtered = stock_meta[
        stock_meta['sector'].isin(selected_sectors) &
        stock_meta['market_cap'].isin(selected_caps)
    ]
    st.sidebar.markdown(f"**{len(filtered)} stocks selected**")
    return filtered

# --- Fetch Adjusted Close Data ---
@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    adj = data['Adj Close'] if 'Adj Close' in data.columns else data
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    return adj.dropna()

# --- Trend Vector Extraction (no caching due to unhashable objects) ---
def compute_trend_vectors(data, window=WINDOW_SIZE):
    trend_vectors = []
    for ticker in data.columns:
        prices = data[ticker].dropna()
        for i in range(len(prices) - window + 1):
            window_prices = prices.iloc[i : i + window].values.reshape(-1, 1)
            scaled = StandardScaler().fit_transform(window_prices).flatten()
            trend_vectors.append({
                'ticker': ticker,
                'start_date': prices.index[i],
                'vector': scaled.tolist(),
            })
    return pd.DataFrame(trend_vectors)

# --- Save / Load Parquet ---
def save_to_parquet(df, path):
    df['vector'] = df['vector'].apply(lambda x: np.array(x))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)

def load_from_parquet(path):
    return pq.read_table(path).to_pandas()

# --- PCA + Clustering ---
def reduce_and_cluster(df, n_clusters=N_CLUSTERS):
    X = np.stack(df['vector'].values)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['pca_1'], df['pca_2'] = X_pca[:, 0], X_pca[:, 1]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_pca)
    return df

# --- Main App ---
def main():
    st.title("📊 F&O Stock Trend Clustering (7-Day Windows)")

    stock_meta = load_stock_metadata()
    filtered_meta = apply_filters(stock_meta)
    TICKERS = filtered_meta['symbol'].tolist()

    if not TICKERS:
        st.warning("No stocks selected. Please adjust filters in sidebar.")
        return

    if not os.path.exists(PARQUET_PATH):
        st.info("Fetching and computing trend vectors (1–2 mins)...")
        raw_data = fetch_data(TICKERS, START_DATE, END_DATE)
        trend_df = compute_trend_vectors(raw_data)
        save_to_parquet(trend_df, PARQUET_PATH)
        st.success("Trend vectors saved.")
    else:
        trend_df = load_from_parquet(PARQUET_PATH)

    trend_df = reduce_and_cluster(trend_df)
    trend_df = trend_df.merge(stock_meta, left_on='ticker', right_on='symbol', how='left')

    st.sidebar.markdown("### Cluster Explorer")
    cluster_id = st.sidebar.selectbox("Select Cluster", sorted(trend_df['cluster'].unique()))
    sample_count = st.sidebar.slider("Samples to Show", 1, 10, 5)

    st.markdown(f"### Cluster {cluster_id} — {len(trend_df[trend_df['cluster']==cluster_id])} windows")
    fig = px.scatter(
        trend_df, x='pca_1', y='pca_2',
        color='sector',
        hover_data=['ticker', 'market_cap', 'start_date'],
        title="Trend Clusters (PCA)",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample Trend Windows")
    cluster_df = trend_df[trend_df['cluster'] == cluster_id].head(sample_count)

    for _, row in cluster_df.iterrows():
        st.caption(f"{row['ticker']} | {row['sector']} | {row['market_cap']} | {row['start_date'].date()}")
        st.line_chart(row['vector'])

if __name__ == "__main__":
    main()
