import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ✅ Step 1: File path
file_path = r"C:\Users\Ritika Keshtwal\OneDrive\Documents\smart_city_sensor.csv"

# ✅ Step 2: Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at: {file_path}")

# ✅ Step 3: Create folder for saving plots
save_dir = r"C:\Users\Ritika Keshtwal\OneDrive\Documents\plots"
os.makedirs(save_dir, exist_ok=True)

# ✅ Step 4: Load dataset
df = pd.read_csv(file_path)
print("Dataset Loaded Successfully ✅")
print("Columns in dataset:", list(df.columns))
print(df.head())

# ✅ Step 5: Select features (numerical columns from dataset)
features = ["Vehicle_Count", "Avg_Speed", "Tx_Size_kB", "Energy_Used_J"]
X = df[features].dropna()

# ✅ Step 6: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Step 7: PCA (reduce to 2D for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ✅ Step 8: Elbow Method (to find best k)
wcss = []
K = range(2, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_pca)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, wcss, 'bo-')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for Optimal k")
plt.savefig(os.path.join(save_dir, "elbow_method.png"))  # ✅ Save graph
plt.show()

# ✅ Step 9: Apply KMeans (k=3 as example)
kmeans = KMeans(n_clusters=3, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_pca)

# ✅ Step 10: Silhouette Score
score = silhouette_score(X_pca, df["Cluster"])
print(f"Silhouette Score: {score:.3f}")

# ✅ Step 11: Final Scatter Plot (Clustering)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster"], cmap="viridis", s=50)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Smart City Sensor Clustering (PCA + KMeans)")
plt.colorbar(label="Cluster")
plt.savefig(os.path.join(save_dir, "cluster_plot.png"))  # ✅ Save graph
plt.show()

print(f"All plots saved in: {save_dir}")

