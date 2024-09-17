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

# Function to define clustering models
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

def spectral_clustering_with_pca(X, n_clusters, n_components, affinity):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0, affinity=affinity)
    return model.fit_predict(X_pca)

# Function to evaluate clustering performance
def evaluate_clustering(X, labels):
    unique_labels = np.unique(labels)
    
    # Handle case of single or no cluster
    if len(unique_labels) < 2:
        st.write("Not enough clusters to compute performance metrics.")
        return None, None, None

    # Compute metrics for non-noise points in DBSCAN
    if -1 in unique_labels:
        mask = labels != -1
        silhouette = silhouette_score(X[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        calinski_harabasz = calinski_harabasz_score(X[mask], labels[mask])
    else:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)

    return silhouette, davies_bouldin, calinski_harabasz

# Function to plot clusters
def plot_3d_clusters(X, labels, title):
    fig = px.scatter_3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2], 
        color=labels.astype(str),
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2', 'z': 'Component 3'}
    )
    st.plotly_chart(fig)

def plot_2d_clusters(X, labels, title):
    fig = px.scatter(
        x=X[:, 0], y=X[:, 1], 
        color=labels.astype(str),
        title=title,
        labels={'x': 'Component 1', 'y': 'Component 2'}
    )
    st.plotly_chart(fig)

def apply_pca_after_clustering(data, labels, algorithm, n_components=3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data)

    # Force 3D visualization for GMM
    if algorithm == "Gaussian Mixture Model (GMM)" and n_components == 3:
        plot_3d_clusters(X_pca, labels, f"PCA 3D Visualization with {n_components} Components and Clusters")
    else:
        plot_2d_clusters(X_pca, labels, f"PCA 2D Visualization with {n_components} Components and Clusters")


# Streamlit App Flow
st.title('Interactive Clustering Dashboard')
st.sidebar.title('Options')

# Step 1: Dataset upload
uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["csv", "xlsx"])

if uploaded_file:
    # Step 2: Load and validate the dataset
    marketing_campaign_data = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('csv') else pd.read_excel(uploaded_file)

    # Pre-selected features for clustering
    selected_vars = [
        'Annual_Income', 'Kidhome', 'Teenhome', 'Recency',
        'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold',
        'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 
        'NumStorePurchases', 'NumWebVisitsMonth'
    ]

    # Ensure the dataset contains the pre-selected features
    missing_features = [var for var in selected_vars if var not in marketing_campaign_data.columns]
    if missing_features:
        st.error(f"The following required features are missing from the dataset: {', '.join(missing_features)}")
    else:
        # Step 3: Drop missing values for the selected features
        X_marketing_campaign = marketing_campaign_data[selected_vars].dropna()

        # Step 4: Allow user to choose clustering algorithm and parameters
        algorithm = st.sidebar.selectbox(
            "Select Clustering Algorithm",
            ["Gaussian Mixture Model (GMM)", "Hierarchical Clustering", "DBSCAN", "Spectral Clustering"]
        )

        # Common inputs
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        n_pca_components = st.sidebar.slider("Number of PCA Components", 2, 3, 2)

        # Algorithm-specific parameters
        if algorithm == "DBSCAN":
            eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
            min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)
        elif algorithm == "Gaussian Mixture Model (GMM)":
            covariance_type = st.sidebar.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])
        elif algorithm == "Hierarchical Clustering":
            linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
        elif algorithm == "Spectral Clustering":
            affinity = st.sidebar.selectbox("Affinity", ["nearest_neighbors", "rbf"])

        # Step 5: Run the clustering algorithm
        st.write(f"Processing data with {algorithm}...")

        if algorithm == "Gaussian Mixture Model (GMM)":
            labels, bic, aic = gmm_clustering(X_marketing_campaign, n_clusters, covariance_type)
            st.write(f"BIC Score: {bic:.4f}, AIC Score: {aic:.4f}")
        elif algorithm == "Hierarchical Clustering":
            labels = hierarchical_clustering(X_marketing_campaign, n_clusters, linkage_method)
            st.subheader("Dendrogram for Hierarchical Clustering")
            Z = linkage(X_marketing_campaign, method=linkage_method)
            fig, ax = plt.subplots()
            dendrogram(Z, ax=ax, truncate_mode='lastp', p=3)
            st.pyplot(fig)
        elif algorithm == "DBSCAN":
            labels = dbscan_clustering(X_marketing_campaign, eps, min_samples)
        elif algorithm == "Spectral Clustering":
            labels = spectral_clustering_with_pca(X_marketing_campaign, n_clusters, n_pca_components, affinity)

        # Step 6: Evaluate clustering performance
        silhouette, db, ch = evaluate_clustering(X_marketing_campaign, labels)

        if silhouette is not None:
            st.write(f"Silhouette Score: {silhouette:.4f}, Davies-Bouldin Score: {db:.4f}, Calinski-Harabasz Score: {ch:.4f}")
        else:
            st.write("Clustering could not be evaluated (e.g., not enough clusters or only noise).")

        # Step 7: PCA Visualization
        st.subheader(f'PCA Visualization with Clustering')
        apply_pca_after_clustering(X_marketing_campaign, labels, algorithm, n_pca_components)

else:
    st.info("Please upload a dataset to start.")
