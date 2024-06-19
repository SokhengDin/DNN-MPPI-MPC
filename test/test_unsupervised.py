import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf

# Generate random data for clustering
X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
X = StandardScaler().fit_transform(X)

# Create and train the models
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

hier = AgglomerativeClustering(n_clusters=4)
hier_labels = hier.fit_predict(X)

dbscan = DBSCAN(eps=0.5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X)

# Variational Autoencoder (VAE)
latent_dim = 2
inputs = Input(shape=(X.shape[1],))
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_h = Dense(64, activation='relu')
decoder_mean = Dense(X.shape[1], activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

vae = Model(inputs, x_decoded_mean)
vae.compile(optimizer=Adam(), loss=MeanSquaredError())
vae.fit(X, X, epochs=50, batch_size=32, verbose=0)

# Plot and save the results
fig, axs = plt.subplots(2, 2, figsize=(16, 14))

# K-Means clustering
axs[0, 0].scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)
axs[0, 0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, linewidths=2, color='red', label='Cluster Centers')
axs[0, 0].set_title('K-Means Clustering', fontsize=16)
axs[0, 0].set_xlabel('Feature 1', fontsize=14)
axs[0, 0].set_ylabel('Feature 2', fontsize=14)
axs[0, 0].legend(fontsize=12)
axs[0, 0].tick_params(axis='both', labelsize=12)

# Hierarchical clustering
axs[0, 1].scatter(X[:, 0], X[:, 1], c=hier_labels, cmap='viridis', alpha=0.7)
axs[0, 1].set_title('Hierarchical Clustering', fontsize=16)
axs[0, 1].set_xlabel('Feature 1', fontsize=14)
axs[0, 1].set_ylabel('Feature 2', fontsize=14)
axs[0, 1].tick_params(axis='both', labelsize=12)

# DBSCAN
unique_labels = set(dbscan_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
        label = 'Noise'
    else:
        label = f'Cluster {k}'
    class_member_mask = dbscan_labels == k
    xy = X[class_member_mask]
    axs[1, 0].plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=8, label=label)
axs[1, 0].set_title('DBSCAN', fontsize=16)
axs[1, 0].set_xlabel('Feature 1', fontsize=14)
axs[1, 0].set_ylabel('Feature 2', fontsize=14)
axs[1, 0].legend(fontsize=12)
axs[1, 0].tick_params(axis='both', labelsize=12)

# Variational Autoencoder (VAE)
encoder = Model(inputs, z_mean)
x_test_encoded = encoder.predict(X)
axs[1, 1].scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.7)
axs[1, 1].set_title('Variational Autoencoder (VAE)', fontsize=16)
axs[1, 1].set_xlabel('Latent Dimension 1', fontsize=14)
axs[1, 1].set_ylabel('Latent Dimension 2', fontsize=14)
axs[1, 1].tick_params(axis='both', labelsize=12)

plt.tight_layout(pad=2.0)
plt.savefig('unsupervised_learning_plots_thesis.png', dpi=300)
plt.show()