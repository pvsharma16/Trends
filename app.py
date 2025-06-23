import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import os
import pyarrow.parquet as pq
import pyarrow as pa

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from hdbscan import HDBSCAN

st.set_page_config(layout="wide")

# --- Constants ---
CSV_PATH = "fno_stocks.csv"
DATA_PATH = "data/trend_vectors.parquet"
WINDOW_SIZE = 7
START_DATE = "2021-06-01"
END_DATE = "2024-06-01"

# --- Load stock metadata ---
@st.cache_data
def load_stock_metadata():
    df = pd.read_csv(CSV_PATH)
    return df.dropna(subset=['symbol'])

# --- Fetch price data ---
@st.cache_data
def fetch_data(tickers):
    data = yf.download(tickers, start=START_DATE, end=END_DATE)
    adj = data['Adj Close'] if 'Adj Close' in data else data
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    return adj.dropna(how='all')

# --- Compute trends ---
def compute_trends(df, window):
    trends = []
    for ticker in df.columns:
        series = df[ticker].dropna()
        if len(series) < window:
            continue
        for i in range(len(series) - window + 1):
            win = series.iloc[i:i+window].values.reshape(-1, 1)
            z = StandardScaler().fit_transform(win).flatten().tolist()
            trends.append({
                "ticker": ticker,
                "start_date": series.index[i],
                "vector": z
            })
    return pd.DataFrame(trends)

# --- Save/Load Parquet ---
def save_to_parquet(df):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df['vector'] = df['vector'].apply(lambda x: np.array(x))
    pq.write_table(pa.Table.from_pandas(df), DATA_PATH)

def load_from_parquet():
    df = pq.read_table(DATA_PATH).to_pandas()
    df['vector'] = df['vector'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    return df

# --- Clustering ---
def cluster_trends(df, algo, n_clusters=5, eps=0.5):
    X = np.stack(df['vector'].values)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['pca_1'], df['pca_2'] = X_pca[:, 0], X_pca[:, 1]

    if algo == "KMeans":
        km = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = km.fit_predict(X_pca)
    elif algo == "DBSCAN":
        db = DBSCAN(eps=eps)
        df['cluster'] = db.fit_predict(X_pca)
    elif algo == "HDBSCAN":
        hdb = HDBSCAN(min_cluster_size=5)
        df['cluster'] = hdb.fit_predict(X_pca)

    return df

# --- Main App ---
def main():
    st.title("📊 F&O Stock Trend Clustering with ML Algorithms")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")
    refresh = st.sidebar.button("🔄 Refresh Yahoo Data")
    algo = st.sidebar.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "HDBSCAN"])
    window = st.sidebar.slider("Trend Window (days)", 5, 14, WINDOW_SIZE)
    n_clusters = st.sidebar.slider("KMeans: Number of Clusters", 2, 10, 5) if algo == "KMeans" else None
    eps = st.sidebar.slider("DBSCAN: Epsilon", 0.1, 1.5, 0.5) if algo == "DBSCAN" else None

    stock_meta = load_stock_metadata()
    sectors = sorted(stock_meta['sector'].dropna().unique())
    caps = sorted(stock_meta['market_cap'].dropna().unique())

    selected_sectors = st.sidebar.multiselect("Sectors", sectors, default=sectors)
    selected_caps = st.sidebar.multiselect("Market Cap", caps, default=caps)

    filtered = stock_meta[
        stock_meta['sector'].isin(selected_sectors) &
        stock_meta['market_cap'].isin(selected_caps)
    ]
    tickers = filtered['symbol'].tolist()

    if not tickers:
        st.warning("No stocks selected.")
        return

    # --- Fetch or refresh data ---
    if refresh or not os.path.exists(DATA_PATH):
        st.info("⏳ Fetching and computing trends...")
        raw = fetch_data(tickers)
        valid = [col for col in raw.columns if raw[col].dropna().shape[0] >= window]
        dropped = list(set(tickers) - set(valid))
        raw = raw[valid]
        if dropped:
            st.sidebar.warning(f"Dropped due to insufficient data: {', '.join(dropped)}")
        trend_df = compute_trends(raw, window)
        save_to_parquet(trend_df)
        st.success("✅ Data updated and saved.")
    else:
        trend_df = load_from_parquet()

    if trend_df.empty:
        st.error("No trends found.")
        return

    st.info("🔬 Running clustering...")
    clustered = cluster_trends(trend_df, algo, n_clusters, eps)
    clustered = clustered.merge(stock_meta, left_on='ticker', right_on='symbol', how='left')

    st.subheader("📍 Cluster Scatter Plot (PCA Reduced)")
    fig = px.scatter(
        clustered, x='pca_1', y='pca_2', color='sector', symbol='cluster',
        hover_data=['ticker', 'market_cap', 'start_date']
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Trend Viewer ---
    st.subheader("🔎 Sample Trends by Cluster")
    valid_clusters = sorted(clustered['cluster'].unique())
    sel = st.selectbox("Select Cluster", valid_clusters)
    sample = clustered[clustered['cluster'] == sel].head(5)
    for _, row in sample.iterrows():
        st.caption(f"{row['ticker']} | {row['sector']} | {row['market_cap']} | {row['start_date'].date()}")
        st.line_chart(row['vector'])

if __name__ == "__main__":
    main()
