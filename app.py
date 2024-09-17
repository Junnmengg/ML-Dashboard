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

# Sidebar option for selecting the clustering algorithm
algorithm = st.sidebar.selectbox(
    "Select Clustering Algorithm",
    ["Gaussian Mixture Model (GMM)", "Hierarchical Clustering", "DBSCAN", "Spectral Clustering"]
)

# User inputs for common parameters
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
n_pca_components = st.sidebar.slider("Number of PCA Components", 2, 3, 2)

# Algorithm-specific parameter input
if algorithm == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)
elif algorithm == "Gaussian Mixture Model (GMM)":
    covariance_type = st.sidebar.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
elif algorithm == "Hierarchical Clustering":
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
elif algorithm == "Spectral Clustering":
    affinity = st.sidebar.selectbox("Affinity", ["nearest_neighbors", "rbf"])

if uploaded_file:
    st.write(f"Processing data with {algorithm}...")

    if algorithm == "Gaussian Mixture Model (GMM)":
        labels, bic, aic = gmm_clustering(X_marketing_campaign, n_clusters, covariance_type)
        st.write(f"BIC Score: {bic:.4f}, AIC Score: {aic:.4f}")
    elif algorithm == "Hierarchical Clustering":
        labels = hierarchical_clustering(X_marketing_campaign, n_clusters, linkage_method)
        # Plot dendrogram for hierarchical clustering
        st.subheader("Dendrogram for Hierarchical Clustering")
        Z = linkage(X_marketing_campaign, method=linkage_method)
        fig, ax = plt.subplots()
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=3)
        st.pyplot(fig)
    elif algorithm == "DBSCAN":
        labels = dbscan_clustering(X_marketing_campaign, eps, min_samples)
    elif algorithm == "Spectral Clustering":
        labels = spectral_clustering_with_pca(X_marketing_campaign, n_clusters, n_pca_components, affinity)

    # Evaluate clustering performance
    silhouette, db, ch = evaluate_clustering(X_marketing_campaign, labels)

    # Display clustering performance metrics if calculated
    if silhouette is not None:
        st.write(f"Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {db:.4f}, Calinski-Harabasz Score: {ch:.4f}")
    else:
        st.write("Clustering could not be evaluated (e.g., not enough clusters or only noise).")

    # Apply PCA after clustering and plot (3D for GMM, 2D for others)
    st.subheader(f'PCA Visualization with Clustering')
    apply_pca_after_clustering(X_marketing_campaign, labels, algorithm, n_pca_components)

# Define clustering models
def gmm_clustering(X, n_clusters, covariance_type):
    gmm = GaussianMixture(n_components=n_clusters, covariance_type=covariance_type, random_state=0)
    labels = gmm.fit_predict(X)
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    return labels, bic, aic

def hierarchical_clustering(X, n_clusters, linkage_method):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X)
    return labels

def dbscan_clustering(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(X)

def spectral_clustering(X, n_clusters, n_components, affinity):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0, affinity=affinity)
    return model.fit_predict(X_pca)
