import numpy as np


def calculate_performance_numpy(distances_matrix, labels):
    """
    For a given distance matrix and labels of all samples, this function calculates two performance measures:
     - The mean CMC scores for n = [1, 3, 5, 10]
     - A mean accuracy metric. This metric calculates how many of the k samples that belong to the same class are among
       the first k ranked elements.

    For N samples, the arguments to this function are:
    :param distances_matrix: A NumPy array defining a distance matrix of floats of size [N, N].
    :param labels: An array of integers of size N.

    """
    assert distances_matrix.shape[0] == distances_matrix.shape[1], "The distance matrix must be a square matrix"
    assert len(labels) == distances_matrix.shape[0], "The size of the matrix should be equal to number of labels"

    # Create a bool matrix (mask) where all the elements are True, except for the diagonal.
    mask = np.logical_not(np.eye(labels.shape[0], dtype=np.bool))

    # Create a bool matrix (label_equal) with value True in the position where the row and column (i, j)
    # belong to the same label, except for i = j.
    label_equal = labels[np.newaxis, :] == labels[:, np.newaxis]
    label_equal = np.logical_and(label_equal, mask)

    # Add the maximum distance to the diagonal.
    distances_matrix = distances_matrix + np.logical_not(mask) * np.max(distances_matrix.flatten(), axis=-1)

    # Get the sorted indices of the distance matrix for each sample.
    sorted_indices = np.argsort(distances_matrix, axis=1)

    # Get a bool matrix where the bool values in label_equal are sorted according to sorted_indices
    sorted_equal_labels_all = np.zeros(label_equal.shape, dtype=bool)
    for i, ri in enumerate(sorted_indices):
        sorted_equal_labels_all[i] = label_equal[i][ri]

    # Calculate the mean CMC scores for k=[1, 3, 5, 10] over all samples
    # The score is 1 if a sample j with the same label as i is in the first k ranked positions. It i s 0 otherwise.
    cmc_scores = np.zeros([4])
    for sorted_equal_labels in sorted_equal_labels_all:
        # CMC scores for a sample
        score = np.asarray([np.sum(sorted_equal_labels[:n]) > 0 for n in [1, 3, 5, 10]])
        # Update running average
        cmc_scores = cmc_scores + score
    cmc_scores /= len(sorted_equal_labels_all)

    # Calculate the accuracy metric

    # Calculate how many samples are there with the same label as any sample i.
    num_positives = np.sum(label_equal, axis=1, dtype=np.int)
    num_samples = len(sorted_equal_labels_all)

    # Calculate the average metric by adding up how many labels correspond to sample i in the first n elements of the
    # ranked row. So, if all the first n elements belong to the same labels the sum is n (perfect score).
    acc = 0
    for i, n in enumerate(num_positives):
        acc = acc + np.sum(sorted_equal_labels_all[i, :n], dtype=np.float32) / (n * num_samples)

    return cmc_scores, acc
