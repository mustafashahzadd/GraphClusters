import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Title and Introduction
st.title("Community Detection in Social Networks Using K-Means")
st.markdown("""
This application demonstrates community detection in social networks using K-Means clustering.
We will use graph theory concepts and tools like NetworkX and Scikit-learn.
""")

# Step 1: Load Dataset
st.header("1. Load Graph Dataset")
dataset = st.selectbox("Choose a dataset", ["Karate Club Network", "Custom Dataset"])

# Load the Karate Club Dataset
if dataset == "Karate Club Network":
    G = nx.karate_club_graph()
    st.write("Loaded the Karate Club Network with ", G.number_of_nodes(), " nodes and ", G.number_of_edges(), " edges.")

# Allow Custom Dataset Upload
elif dataset == "Custom Dataset":
    uploaded_file = st.file_uploader("Upload your dataset (CSV with edges)", type=["csv"])
    if uploaded_file is not None:
        edges_df = pd.read_csv(uploaded_file)
        G = nx.from_pandas_edgelist(edges_df, source='source', target='target')
        st.write("Loaded custom graph with ", G.number_of_nodes(), " nodes and ", G.number_of_edges(), " edges.")
    else:
        st.warning("Please upload a dataset to proceed.")

# Ensure graph is loaded before proceeding
if 'G' in locals():
    # Step 2: Feature Extraction
    st.header("2. Feature Extraction")
    st.markdown("Features are extracted from the graph for clustering (e.g., degree, centrality measures).")
    features = pd.DataFrame({
        'Node': list(G.nodes),
        'Degree': [val for (node, val) in G.degree()],
        'Closeness': [nx.closeness_centrality(G)[node] for node in G.nodes],
        'Betweenness': [nx.betweenness_centrality(G)[node] for node in G.nodes]
    })
    st.write(features)

    # Step 3: Apply K-Means Clustering
    st.header("3. Apply K-Means Clustering")
    num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(features[['Degree', 'Closeness', 'Betweenness']])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    features['Cluster'] = clusters
    st.write("Clustering Results", features)

    # Step 4: Visualize the Communities
    st.header("4. Visualize the Communities")
    fig, ax = plt.subplots(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=False,
        node_color=[clusters[list(G.nodes).index(node)] for node in G.nodes()],
        cmap=plt.cm.Set1, ax=ax, node_size=50
    )
    st.pyplot(fig)

    # Step 5: Evaluate Performance
    st.header("5. Evaluate Performance")
    st.markdown("""
    We evaluate clustering using modularity, silhouette score, and coverage.
    """)

    # Modularity
    modularity = nx.algorithms.community.modularity(
        G, [list(features[features['Cluster'] == i]['Node']) for i in range(num_clusters)]
    )
    st.write("Modularity Score: ", modularity)

    # Silhouette Score
    silhouette = silhouette_score(X, clusters)
    st.write("Silhouette Score: ", silhouette)

    # Coverage
    intra_edges = 0
    total_edges = len(G.edges)
    for u, v in G.edges:
        if clusters[list(G.nodes).index(u)] == clusters[list(G.nodes).index(v)]:
            intra_edges += 1
    coverage = intra_edges / total_edges
    st.write("Coverage: ", coverage)

    st.markdown("### Conclusion")
    st.markdown("""
    This tool demonstrates how graph-based features and K-Means clustering can be used for community detection. 
    Modularity measures the quality of community structure, Silhouette Score evaluates the separation and cohesion of clusters, 
    and Coverage measures the fraction of edges correctly assigned within clusters.
    """)
else:
    st.warning("Please load a graph dataset to proceed.")
