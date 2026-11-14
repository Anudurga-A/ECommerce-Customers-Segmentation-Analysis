
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print(" E-COMMERCE CUSTOMER SEGMENTATION AND RECOMMENDATION ")
print("=" * 60)

os.makedirs("output/graphs", exist_ok=True)

# 1. LOAD DATASET

print("\nLoading Dataset...")

df = pd.read_excel("dataset/OnlineRetail.xlsx")

print("Dataset Loaded Successfully!")
print("\nFIRST 5 TRANSACTION RECORDS")
print(df.head())

print("\nDATASET INFORMATION")
df.info()

print("\nDATASET SHAPE")
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE RECORDS")
print("Total Duplicates:", df.duplicated().sum())

# 2. DATA CLEANING

print("\nCleaning Dataset...")

df = df.drop_duplicates()
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

df["CustomerID"] = df["CustomerID"].astype(int)

print("Data Cleaning Completed!")
print("Cleaned Transactions:", len(df))

# 3. FEATURE ENGINEERING

print("\nPerforming Feature Engineering...")

df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

customer_data = df.groupby("CustomerID").agg(
    TotalOrders=("InvoiceNo", "nunique"),
    TotalQuantity=("Quantity", "sum"),
    TotalSpending=("TotalAmount", "sum"),
    AverageOrderValue=("TotalAmount", "mean")
).reset_index()

customer_data.to_csv(
    "output/customer_feature_table.csv", index=False
)

print("\nCUSTOMER LEVEL FEATURE TABLE")
print(customer_data.head())

# 4. RFM ANALYSIS

print("\nPerforming RFM Analysis...")

reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalAmount", "sum")
).reset_index()

rfm.to_csv("output/rfm_analysis.csv", index=False)

print("\nRFM ANALYSIS RESULTS")
print(rfm.head())

# 5. FEATURE SCALING USING NUMPY

print("\nPerforming Feature Scaling...")

features = rfm[["Recency", "Frequency", "Monetary"]].to_numpy(dtype=float)

feature_mean = np.mean(features, axis=0)
feature_std = np.std(features, axis=0)

feature_std[feature_std == 0] = 1

scaled_features = (features - feature_mean) / feature_std

print("Feature Scaling Completed!")

# 6. K-MEANS ALGORITHM USING NUMPY

def perform_kmeans(data, k, max_iter=100, random_state=42):

    rng = np.random.default_rng(random_state)

    best_inertia = float("inf")
    best_labels = None
    best_centers = None

    for restart in range(5):

        # K-Means++ initialization

        centers = [data[rng.integers(len(data))]]

        for _ in range(1, k):

            distances = np.min(
                np.sum(
                    (data[:, np.newaxis, :] - np.array(centers)[np.newaxis, :, :]) ** 2,
                    axis=2
                ),
                axis=1
            )

            total_distance = distances.sum()

            if total_distance == 0:
                next_index = rng.integers(len(data))
            else:
                probabilities = distances / total_distance
                next_index = rng.choice(len(data), p=probabilities)

            centers.append(data[next_index])

        centers = np.array(centers, dtype=float)

        # Iterative clustering

        for iteration in range(max_iter):

            distances = np.sum(
                (data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
                axis=2
            )

            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()

            for cluster_id in range(k):

                cluster_points = data[labels == cluster_id]

                if len(cluster_points) > 0:
                    new_centers[cluster_id] = np.mean(
                        cluster_points, axis=0
                    )

            if np.allclose(centers, new_centers):
                centers = new_centers
                break

            centers = new_centers

        final_distances = np.sum(
            (data - centers[labels]) ** 2,
            axis=1
        )

        inertia = np.sum(final_distances)

        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, best_inertia


# 7. ELBOW METHOD

print("\nCalculating Elbow Method...")

inertia_values = []

for k in range(2, 11):

    labels, centers, inertia = perform_kmeans(
        scaled_features,
        k
    )

    inertia_values.append(inertia)

    print("Clusters:", k, "| Inertia:", round(inertia, 2))

plt.figure(figsize=(9, 6))
plt.plot(range(2, 11), inertia_values, marker="o")
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig("output/graphs/elbow_method.png")
plt.close()

print("Elbow Method Graph Saved!")

# 8. FINAL K-MEANS CLUSTERING

print("\nPerforming Final K-Means Clustering...")

cluster_labels, cluster_centers, final_inertia = perform_kmeans(
    scaled_features,
    4
)

rfm["Cluster"] = cluster_labels

print("K-Means Clustering Completed!")

# 9. SILHOUETTE SCORE

print("\nCalculating Silhouette Score...")

def calculate_silhouette(data, labels, sample_size=500):

    rng = np.random.default_rng(42)

    sample_size = min(sample_size, len(data))
    sample_indices = rng.choice(
        len(data), size=sample_size, replace=False
    )

    sample_data = data[sample_indices]
    sample_labels = labels[sample_indices]

    distance_matrix = np.sqrt(
        np.sum(
            (sample_data[:, np.newaxis, :] -
             data[np.newaxis, :, :]) ** 2,
            axis=2
        )
    )

    unique_clusters = np.unique(labels)
    scores = []

    for i in range(sample_size):

        own_cluster = sample_labels[i]

        own_mask = labels == own_cluster

        own_distances = distance_matrix[i, own_mask]

        a = own_distances.mean() if len(own_distances) > 1 else 0

        other_distances = []

        for cluster_id in unique_clusters:

            if cluster_id != own_cluster:

                cluster_mask = labels == cluster_id

                mean_distance = distance_matrix[
                    i, cluster_mask
                ].mean()

                other_distances.append(mean_distance)

        if not other_distances:
            continue

        b = min(other_distances)

        denominator = max(a, b)

        if denominator == 0:
            scores.append(0)
        else:
            scores.append((b - a) / denominator)

    return float(np.mean(scores))


score = calculate_silhouette(
    scaled_features,
    cluster_labels
)

print("Silhouette Score:", round(score, 4))

# 10. CLUSTER ANALYSIS

print("\nCUSTOMER CLUSTER DISTRIBUTION")

print(
    rfm["Cluster"].value_counts().sort_index()
)

cluster_summary = rfm.groupby("Cluster").agg(
    TotalCustomers=("CustomerID", "count"),
    AverageRecency=("Recency", "mean"),
    AverageFrequency=("Frequency", "mean"),
    AverageSpending=("Monetary", "mean")
).round(2)

print("\nCLUSTER SUMMARY")
print(cluster_summary)

rfm.to_csv(
    "output/customer_segments.csv", index=False
)

cluster_summary.to_csv(
    "output/cluster_summary.csv"
)

# 11. CUSTOMER CLUSTER VISUALIZATION

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="viridis",
    s=50
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Purchase Frequency")
plt.ylabel("Total Spending")
plt.tight_layout()

plt.savefig(
    "output/graphs/customer_clusters.png"
)

plt.close()

print("\nCustomer Cluster Graph Saved!")

# 12. CUSTOMER SPENDING GRAPH

plt.figure(figsize=(10, 6))

sns.histplot(
    data=rfm,
    x="Monetary",
    bins=40,
    kde=True
)

plt.title("Customer Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")
plt.tight_layout()

plt.savefig(
    "output/graphs/customer_spending.png"
)

plt.close()

# 13. RFM CORRELATION GRAPH

plt.figure(figsize=(8, 6))

sns.heatmap(
    rfm[["Recency", "Frequency", "Monetary"]].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("RFM Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "output/graphs/rfm_correlation.png"
)

plt.close()

# 14. FINAL RESULTS

print("\n" + "=" * 60)
print(" CUSTOMER SEGMENTATION PROJECT COMPLETED ")
print("=" * 60)

print("Total Customers:", rfm["CustomerID"].nunique())
print("Total Revenue:", round(df["TotalAmount"].sum(), 2))
print("Total Clusters:", rfm["Cluster"].nunique())
print("Silhouette Score:", round(score, 4))

print("\nGENERATED FILES:")

print("1. output/customer_feature_table.csv")
print("2. output/rfm_analysis.csv")
print("3. output/customer_segments.csv")
print("4. output/cluster_summary.csv")
print("5. output/graphs/customer_spending.png")
print("6. output/graphs/rfm_correlation.png")
print("7. output/graphs/elbow_method.png")
print("8. output/graphs/customer_clusters.png")

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
