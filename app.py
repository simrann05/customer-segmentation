import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("Mall Customer Segmentation using KMeans Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded_file:
    customer_data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(customer_data.head())

    # Shape and info
    st.write(f"**Shape of dataset:** {customer_data.shape}")
    
    if customer_data.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. Please handle them before proceeding.")
    else:
        X = customer_data.iloc[:, [3, 4]].values  # Annual Income and Spending Score

        # Elbow Method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        st.subheader("Elbow Method to Determine Optimal Number of Clusters")
        fig1, ax1 = plt.subplots()
        sns.set()
        ax1.plot(range(1, 11), wcss, marker='o')
        ax1.set_title("The Elbow Point Graph")
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("WCSS")
        st.pyplot(fig1)

        # Choose number of clusters
        clusters = st.slider("Select number of clusters (as per elbow graph)", 2, 10, 5)

        # Apply KMeans
        kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
        Y = kmeans.fit_predict(X)

        st.subheader("Customer Segments Visualized")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        colors = ['green', 'red', 'violet', 'blue', 'yellow', 'orange', 'purple', 'brown', 'pink', 'grey']

        for i in range(clusters):
            ax2.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f'Cluster {i+1}')

        # Centroids
        ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                    s=100, c='cyan', label='Centroids', marker='X')

        ax2.set_title("Customer Groups")
        ax2.set_xlabel("Annual Income (k$)")
        ax2.set_ylabel("Spending Score (1-100)")
        ax2.legend()
        st.pyplot(fig2)

else:
    st.info("Please upload the 'Mall_Customers.csv' file to begin.")
