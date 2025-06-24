import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

#paths to the files
train_images_path = "train-images.idx3-ubyte"
train_labels_path = "train-labels.idx1-ubyte"
test_images_path = "t10k-images.idx3-ubyte"
test_labels_path = "t10k-labels.idx1-ubyte"

#to load and normalize MNIST images
def load_mnist_images(filename):
    images = idx2numpy.convert_from_file(filename)
    images = images.astype(np.float32) / 255.0
    return images.reshape(images.shape[0], -1)

#to load MNIST labels
def load_mnist_labels(filename):
    return idx2numpy.convert_from_file(filename)

#to filter images of the digits 0,1,2 by considering them as classes
def filter_classes(images, labels, classes=[0, 1, 2]):
    mask = np.isin(labels, classes)
    return images[mask], labels[mask]

#sampling equal number of images for each selected class
def sample_n_per_class(images, labels, n=100):
    selected_images = []
    selected_labels = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        selected_idx = np.random.choice(idx, n, replace=False)
        selected_images.append(images[selected_idx])
        selected_labels.append(labels[selected_idx])
    return np.vstack(selected_images), np.hstack(selected_labels)

#lda training
def lda_fit(X, y):
    classes = np.unique(y)
    means = {}
    cov = np.zeros((X.shape[1], X.shape[1]))
    priors = {}
    for c in classes:
        X_c = X[y == c]
        means[c] = np.mean(X_c, axis=0)
        cov += np.cov(X_c, rowvar=False) * (len(X_c) - 1)
        priors[c] = len(X_c) / len(y)
    cov /= len(y) - len(classes)
    return means, cov, priors

#lda prediction
def lda_predict(X, means, cov, priors):
    cov_inv = np.linalg.pinv(cov)
    predictions = []
    for x in X:
        scores = {c: -0.5 * np.dot(np.dot((x - means[c]).T, cov_inv), (x - means[c])) + np.log(priors[c]) for c in means}
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

#qda training
def qda_fit(X, y):
    classes = np.unique(y)
    means = {}
    covs = {}
    priors = {}
    for c in classes:
        X_c = X[y == c]
        means[c] = np.mean(X_c, axis=0)
        covs[c] = np.cov(X_c, rowvar=False)
        priors[c] = len(X_c) / len(y)
    return means, covs, priors

#qda prediction
def qda_predict(X, means, covs, priors):
    predictions = []
    for x in X:
        scores = {}
        for c in means:
            cov_inv = np.linalg.pinv(covs[c])
            det_cov = np.linalg.det(covs[c])
            exponent = -0.5 * np.dot(np.dot((x - means[c]).T, cov_inv), (x - means[c]))
            scores[c] = exponent - 0.5 * np.log(det_cov) + np.log(priors[c])
        predictions.append(max(scores, key=scores.get))
    return np.array(predictions)

#perform PCA and return reduced images based on variance threshold
def perform_pca(train_images, test_images, variance_threshold=0.95):
    mean_train = np.mean(train_images, axis=0)
    Xc = train_images - mean_train
    N = Xc.shape[0]  # Number of samples
    cov_matrix = np.zeros((Xc.shape[1], Xc.shape[1]))
    for i in range(N):
        cov_matrix += np.outer(Xc[i], Xc[i])
    cov_matrix /= (N - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    Up = eigenvectors[:, :k]
    train_images_pca = Xc @ Up
    test_images_pca = (test_images - mean_train) @ Up
    return train_images_pca, test_images_pca

#fda implementation using PCA results
def perform_fda(train_images_pca, test_images_pca, train_labels):
    num_classes = len(np.unique(train_labels))
    overall_mean = np.mean(train_images_pca, axis=0)
    S_B = np.zeros((train_images_pca.shape[1], train_images_pca.shape[1]))
    S_W = np.zeros((train_images_pca.shape[1], train_images_pca.shape[1]))

    for c in np.unique(train_labels):
        X_c = train_images_pca[train_labels == c]
        mean_c = np.mean(X_c, axis=0)
        S_B += len(X_c) * np.outer(mean_c - overall_mean, mean_c - overall_mean)
        S_W += np.cov(X_c, rowvar=False) * (len(X_c) - 1)

    S_W_inv = np.linalg.pinv(S_W)
    eigenvalues_fda, eigenvectors_fda = np.linalg.eigh(S_W_inv @ S_B)
    sorted_indices_fda = np.argsort(eigenvalues_fda)[::-1]
    eigenvectors_fda = eigenvectors_fda[:, sorted_indices_fda]
    k_fda = num_classes - 1
    W_fda = eigenvectors_fda[:, :k_fda]
    train_images_fda = train_images_pca @ W_fda
    test_images_fda = test_images_pca @ W_fda

    return train_images_fda, test_images_fda

#to plot transformed feature space
def plot_transformed_space(data, labels, title):
    plt.figure(figsize=(8, 6))
    for c, marker, color in zip([0, 1, 2], ['o', 's', '^'], ['r', 'g', 'b']):
        plt.scatter(data[labels == c, 0], data[labels == c, 1], marker=marker, color=color, label=f'Class {c}', alpha=0.7)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

#loading training and testing datasets
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

#filtering dataset to contain only the selected items
train_images, train_labels = filter_classes(train_images, train_labels)
test_images, test_labels = filter_classes(test_images, test_labels)

#filter dataset to only contain selected classes
train_images, train_labels = sample_n_per_class(train_images, train_labels, n=100)
test_images, test_labels = sample_n_per_class(test_images, test_labels, n=100)

#save the processed dataset
np.save("train_images.npy", train_images)
np.save("train_labels.npy", train_labels)
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)

# --- Part 1 ---
train_images_pca_95, test_images_pca_95 = perform_pca(train_images, test_images, variance_threshold=0.95)
train_images_fda_95, test_images_fda_95 = perform_fda(train_images_pca_95, test_images_pca_95, train_labels)

lda_means_95, lda_cov_95, lda_priors_95 = lda_fit(train_images_fda_95, train_labels)
train_preds_lda_95 = lda_predict(train_images_fda_95, lda_means_95, lda_cov_95, lda_priors_95)
test_preds_lda_95 = lda_predict(test_images_fda_95, lda_means_95, lda_cov_95, lda_priors_95)
lda_train_acc_95 = np.mean(train_preds_lda_95 == train_labels)
lda_test_acc_95 = np.mean(test_preds_lda_95 == test_labels)

qda_means_95, qda_covs_95, qda_priors_95 = qda_fit(train_images_fda_95, train_labels)
train_preds_qda_95 = qda_predict(train_images_fda_95, qda_means_95, qda_covs_95, qda_priors_95)
test_preds_qda_95 = qda_predict(test_images_fda_95, qda_means_95, qda_covs_95, qda_priors_95)
qda_train_acc_95 = np.mean(train_preds_qda_95 == train_labels)
qda_test_acc_95 = np.mean(test_preds_qda_95 == test_labels)

print("\nPart 1: ")
print(f"FDA + LDA (95% variance) - Train: {lda_train_acc_95:.4f}, Test: {lda_test_acc_95:.4f}")
print(f"FDA + QDA (95% variance) - Train: {qda_train_acc_95:.4f}, Test: {qda_test_acc_95:.4f}")

# --- Part 2 ---
train_images_pca_95, test_images_pca_95 = perform_pca(train_images, test_images, variance_threshold=0.95)
lda_means_95, lda_cov_95, lda_priors_95 = lda_fit(train_images_pca_95, train_labels)
train_preds_lda_95 = lda_predict(train_images_pca_95, lda_means_95, lda_cov_95, lda_priors_95)
test_preds_lda_95 = lda_predict(test_images_pca_95, lda_means_95, lda_cov_95, lda_priors_95)
lda_train_acc_95 = np.mean(train_preds_lda_95 == train_labels)
lda_test_acc_95 = np.mean(test_preds_lda_95 == test_labels)

print("\nPart 2: ")
print(f"PCA + LDA (95% variance) - Train: {lda_train_acc_95:.4f}, Test: {lda_test_acc_95:.4f}")

# --- Part 3 ---
train_images_pca_90, test_images_pca_90 = perform_pca(train_images, test_images, variance_threshold=0.90)
train_images_fda_90, test_images_fda_90 = perform_fda(train_images_pca_90, test_images_pca_90, train_labels)

lda_means_90, lda_cov_90, lda_priors_90 = lda_fit(train_images_fda_90, train_labels)
train_preds_lda_90 = lda_predict(train_images_fda_90, lda_means_90, lda_cov_90, lda_priors_90)
test_preds_lda_90 = lda_predict(test_images_fda_90, lda_means_90, lda_cov_90, lda_priors_90)
lda_train_acc_90 = np.mean(train_preds_lda_90 == train_labels)
lda_test_acc_90 = np.mean(test_preds_lda_90 == test_labels)

qda_means_90, qda_covs_90, qda_priors_90 = qda_fit(train_images_fda_90, train_labels)
train_preds_qda_90 = qda_predict(train_images_fda_90, qda_means_90, qda_covs_90, qda_priors_90)
test_preds_qda_90 = qda_predict(test_images_fda_90, qda_means_90, qda_covs_90, qda_priors_90)
qda_train_acc_90 = np.mean(train_preds_qda_90 == train_labels)
qda_test_acc_90 = np.mean(test_preds_qda_90 == test_labels)

print("\nPart 3: ")
print(f"PCA + FDA + LDA (90% variance) - Train: {lda_train_acc_90:.4f}, Test: {lda_test_acc_90:.4f}")
print(f"PCA + FDA + QDA (90% variance) - Train: {qda_train_acc_90:.4f}, Test: {qda_test_acc_90:.4f}")

# --- Part 4 ---
train_images_pca_2, test_images_pca_2 = perform_pca(train_images, test_images, variance_threshold=1.0)
train_images_fda_2, test_images_fda_2 = perform_fda(train_images_pca_2, test_images_pca_2, train_labels)

lda_means_2, lda_cov_2, lda_priors_2 = lda_fit(train_images_fda_2, train_labels)
train_preds_lda_2 = lda_predict(train_images_fda_2, lda_means_2, lda_cov_2, lda_priors_2)
test_preds_lda_2 = lda_predict(test_images_fda_2, lda_means_2, lda_cov_2, lda_priors_2)
lda_train_acc_2 = np.mean(train_preds_lda_2 == train_labels)
lda_test_acc_2 = np.mean(test_preds_lda_2 == test_labels)

qda_means_2, qda_covs_2, qda_priors_2 = qda_fit(train_images_fda_2, train_labels)
train_preds_qda_2 = qda_predict(train_images_fda_2, qda_means_2, qda_covs_2, qda_priors_2)
test_preds_qda_2 = qda_predict(test_images_fda_2, qda_means_2, qda_covs_2, qda_priors_2)
qda_train_acc_2 = np.mean(train_preds_qda_2 == train_labels)
qda_test_acc_2 = np.mean(test_preds_qda_2 == test_labels)

print("\nPart 4: ")
print(f"PCA + FDA + LDA (First 2 Components) - Train: {lda_train_acc_2:.4f}, Test: {lda_test_acc_2:.4f}")
print(f"PCA + FDA + QDA (First 2 Components) - Train: {qda_train_acc_2:.4f}, Test: {qda_test_acc_2:.4f}")

#plotting PCA-transformed space
plot_transformed_space(train_images_pca_95, train_labels, "PCA - 2D Projection")

#plotting FDA-transformed space
plot_transformed_space(train_images_fda_95, train_labels, "FDA - 2D Projection")
