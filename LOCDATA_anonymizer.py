import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import folium
import hashlib

# Generate sample data with names
# Adjusted coordinates to reflect Tabriz city's geographical bounds
data = pd.DataFrame({
    'latitude': np.random.normal(38.08, 0.05, 100),  # Mean at 38.08, std deviation 0.05
    'longitude': np.random.normal(46.29, 0.05, 100),  # Mean at 46.29, std deviation 0.05
    'name': [f'Person_{i}' for i in range(100)]  # Sample names
})

# Function to add Laplace noise
def add_laplace_noise(data, epsilon):
    scale = 1 / epsilon
    noisy_data = data.copy()
    noisy_data['latitude'] += np.random.laplace(0, scale, size=len(data))
    noisy_data['longitude'] += np.random.laplace(0, scale, size=len(data))
    return noisy_data

# Function to anonymize names using hash function
def anonymize_names(names):
    return names.apply(lambda x: hashlib.sha256(x.encode()).hexdigest())

# Function to calculate displacement due to noise
def calculate_displacement(original_data, noisy_data):
    distances = np.sqrt((original_data['latitude'] - noisy_data['latitude'])**2 + (original_data['longitude'] - noisy_data['longitude'])**2)
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    std_distance = np.std(distances)
    return avg_distance, max_distance, std_distance

# Function to perform DBSCAN clustering
def perform_clustering(data, eps, min_samples):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    return clustering.labels_

# Function to evaluate clustering
def evaluate_clustering(data, labels):
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
    else:
        silhouette = -1  # Invalid silhouette score for a single cluster
        davies_bouldin = -1  # Invalid Davies-Bouldin score for a single cluster
    return silhouette, davies_bouldin

# Adjust the epsilon value here
epsilon = float(input("Enter the epsilon value (privacy parameter): "))

# Apply the function to add noise to the data
noisy_data = add_laplace_noise(data, epsilon)

# Anonymize the names in the noisy data
noisy_data['name'] = anonymize_names(noisy_data['name'])

# Calculate and print displacement metrics
avg_displacement, max_displacement, std_displacement = calculate_displacement(data, noisy_data)
print(f"Average displacement due to noise: {avg_displacement}")
print(f"Maximum displacement due to noise: {max_displacement}")
print(f"Standard deviation of displacement due to noise: {std_displacement}")

# Perform clustering on original and noisy data
original_clusters = perform_clustering(data[['latitude', 'longitude']], eps=0.01, min_samples=3)
noisy_clusters = perform_clustering(noisy_data[['latitude', 'longitude']], eps=0.01, min_samples=3)

# Evaluate clustering quality
original_silhouette, original_davies_bouldin = evaluate_clustering(data[['latitude', 'longitude']], original_clusters)
noisy_silhouette, noisy_davies_bouldin = evaluate_clustering(noisy_data[['latitude', 'longitude']], noisy_clusters)

print(f"Original data - Silhouette Score: {original_silhouette}, Davies-Bouldin Index: {original_davies_bouldin}")
print(f"Noisy data - Silhouette Score: {noisy_silhouette}, Davies-Bouldin Index: {noisy_davies_bouldin}")

# Visualize original and noisy data on a map using folium
map_original = folium.Map(location=[38.08, 46.29], zoom_start=13)
map_noisy = folium.Map(location=[38.08, 46.29], zoom_start=13)

for _, row in data.iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=3, color='blue', fill=True, popup=row['name']).add_to(map_original)

for _, row in noisy_data.iterrows():
    folium.CircleMarker(location=[row['latitude'], row['longitude']], radius=3, color='red', fill=True, popup=row['name']).add_to(map_noisy)

# Save maps to HTML files
map_original.save('original_map.html')
map_noisy.save('noisy_map.html')

# Visualize original and noisy data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data['longitude'], data['latitude'], c=original_clusters, cmap='viridis', alpha=0.5, label='Original Data')
plt.title('Original Geospatial Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(noisy_data['longitude'], noisy_data['latitude'], c=noisy_clusters, cmap='viridis', alpha=0.5, label='Noisy Data')
plt.title('Anonymized Geospatial Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

plt.tight_layout()
plt.show()

# Print the URLs for the interactive maps
print("Original data map saved as original_map.html")
print("Noisy data map saved as noisy_map.html")
