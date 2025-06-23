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

# --- Constants ---
CSV_PATH = 'fno_stocks.csv'
PARQUET_PATH = 'data/trend_vectors.parquet'
START_DATE = '2021-06-01'
END_DATE = '2024-06-01'
WINDOW_SIZE = 7
N_CLUSTERS = 5

# --- Load CSV with sector & market cap ---
@st.cache_data
def load_stock_metadata(path=CSV_PATH):
    df = pd.read_csv(path)
    df.dropna(subset=['symbol'], inplace=True)
    return df

# --- Filter Sidebar ---
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

# --- Fetch stock data ---
@st.cache_data
def fetch_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    
    if 'Adj Close' in data.columns:
        adj_close = data['Adj Close']
    else:
        adj_close = data  # fallback

    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()

    return adj_close.dropna()

# --- Trend vector extraction ---
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

# --- Save & Load Parquet ---
def save_to_parquet(df, path):
    df['vector'] = df['vector'].apply(lambda x: np.array(x))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), path)

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

# --- Main Streamlit App ---
def main():
    st.title("📊 F&O Stock Trend Clustering (7-Day Windows)")

    # Load and filter metadata
    stock_meta = load_stock_metadata()
    filtered_meta = apply_filters(stock_meta)
    TICKERS = filtered_meta['symbol'].tolist()

    if len(TICKERS) == 0:
        st.warning("No stocks selected. Please adjust your filters.")
        return

    # Check for saved data
    if not os.path.exists(PARQUET_PATH):
        st.info("Fetching and processing data. This takes a minute...")
        raw_data = fetch_data(TICKERS, START_DATE, END_DATE)
        trend_df = compute_trend_vectors(raw_data, WINDOW_SIZE)
        save_to_parquet(trend_df, PARQUET_PATH)
        st.success("Trend vectors saved for future use.")
    else:
        trend_df = load_from_parquet(PARQUET_PATH)

    # Reduce and cluster
    trend_df = reduce_and_cluster(trend_df, N_CLUSTERS)

    # Merge metadata
    trend_df = trend_df.merge(stock_meta, left_on='ticker', right_on='symbol', how='left')

    # Cluster selection UI
    st.sidebar.markdown("### Cluster Explorer")
    selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(trend_df['cluster'].unique()))
    num_samples = st.sidebar.slider("Samples to display", 1, 10, 5)

    # Show plot
    st.markdown(f"### Cluster {selected_cluster} — {len(trend_df[trend_df['cluster'] == selected_cluster])} windows")
    fig = px.scatter(
        trend_df, x='pca_1', y='pca_2',
        color='sector',
        hover_data=['ticker', 'market_cap', 'start_date'],
        title="Trend Cluster Map (PCA)",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show trend previews
    st.subheader("Sample Trend Shapes")
    sample_df = trend_df[trend_df['cluster'] == selected_cluster].head(num_samples)

    for _, row in sample_df.iterrows():
        st.caption(f"{row['ticker']} | {row['sector']} | {row['market_cap']} | {row['start_date'].date()}")
        st.line_chart(row['vector'])

if __name__ == "__main__":
    main()
