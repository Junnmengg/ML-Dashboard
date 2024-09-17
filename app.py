import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.graph_objs import Scatter3d, Layout
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Set page title and layout
st.title('Interactive Clustering Dashboard')
st.sidebar.title('Options')

# Define the selected variables for clustering
selected_vars = [
    'Annual_Income', 'Kidhome', 'Teenhome', 'Recency',
    'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
]

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

# Check if the dataset is uploaded
if uploaded_file:
    
    marketing_campaign_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)

    # Select only the chosen columns
    X_marketing_campaign = marketing_campaign_data[selected_vars].dropna()

    st.write("Data Overview:")
    st.write(X_marketing_campaign.head())
else:
    st.info("Please upload a dataset")

# Function for interactive data exploration
def explore_data(data):
    st.subheader("Data Exploration")
    st.write(data)
    st.dataframe(data.describe())
    st.write("Select columns to visualize:")
    columns = st.multiselect("Columns", data.columns)
    if columns:
        fig = px.scatter_matrix(data[columns], title="Scatter Matrix Plot")
        st.plotly_chart(fig)

# Function to plot 3D clusters with white background
def plot_3d_clusters(X, labels, title):
    fig = px.scatter_3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], 
        color=labels.astype(str),
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'}
    )
    st.plotly_chart(fig)

# Function to plot 2D clusters
def plot_2d_clusters(X, labels, title):
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1], 
        color=labels.astype(str),
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2'}
    )
    st.plotly_chart(fig)

# Function to apply PCA after clustering
def apply_pca_after_clustering(data, labels, algorithm, n_components=3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data)

    if algorithm == "Gaussian Mixture Model (GMM)":
        plot_3d_clusters(X_pca, labels, f"PCA 3D Visualization with {n_components} Components and Clusters")
    else:
        plot_2d_clusters(X_pca, labels, f"PCA 2D Visualization with {n_components} Components and Clusters")

# Function to evaluate clustering performance
def evaluate_clustering(X, labels):
    unique_labels = np.unique(labels)
    
    # Handle cases where there is only one or no cluster
    if len(unique_labels) < 2:
        st.write("Not enough clusters to compute performance metrics.")
        return None, None, None

    # Compute metrics only for non-noise points in DBSCAN
    if -1 in unique_labels:
        mask = labels != -1  # Ignore noise points (labeled as -1)
        silhouette = silhouette_score(X[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        calinski_harabasz = calinski_harabasz_score(X[mask], labels[mask])
    else:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)

    return silhouette, davies_bouldin, calinski_harabasz

if uploaded_file:
    # Define clustering models
    def gmm_clustering(X, n_clusters=3):
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        return labels, bic, aic

    def hierarchical_clustering(X, n_clusters=3):
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        return labels

    def dbscan_clustering(X, eps=0.5, min_samples=5):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        return dbscan.fit_predict(X)

    # Spectral Clustering with PCA
    def spectral_clustering_with_pca(X, n_clusters=3, n_components=10):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0, affinity='nearest_neighbors')
        return model.fit_predict(X_pca)

    # Sidebar option for selecting the clustering algorithm
    algorithm = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["Gaussian Mixture Model (GMM)", "Hierarchical Clustering", "DBSCAN", "Spectral Clustering"]
    )

    # Perform clustering based on selected algorithm
    st.write(f"Processing data with {algorithm}...")

    if algorithm == "Gaussian Mixture Model (GMM)":
        labels, bic, aic = gmm_clustering(X_marketing_campaign)
        st.write(f"BIC Score: {bic:.4f}, AIC Score: {aic:.4f}")
    elif algorithm == "Hierarchical Clustering":
        labels = hierarchical_clustering(X_marketing_campaign)
        # Plot dendrogram for hierarchical clustering
        st.subheader("Dendrogram for Hierarchical Clustering")
        Z = linkage(X_marketing_campaign, 'ward')
        fig, ax = plt.subplots()
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=3)
        st.pyplot(fig)
    elif algorithm == "DBSCAN":
        labels = dbscan_clustering(X_marketing_campaign)
    elif algorithm == "Spectral Clustering":
        labels = spectral_clustering_with_pca(X_marketing_campaign)

    # Evaluate clustering performance
    silhouette, db, ch = evaluate_clustering(X_marketing_campaign, labels)

    # Display evaluation results only if metrics are computed
    if silhouette is not None:
        st.write(f"Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {db:.4f}, Calinski-Harabasz Score: {ch:.4f}")
    else:
        st.write("Clustering could not be evaluated (e.g., not enough clusters or only noise).")

    # Apply PCA after clustering and plot accordingly (3D for GMM, 2D for others)
    st.subheader(f'PCA Visualization with Clustering')
    apply_pca_after_clustering(X_marketing_campaign, labels, algorithm)
