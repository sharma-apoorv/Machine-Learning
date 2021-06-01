import numpy as np

from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

def pre_process_data(data, feature_names=None, categories=None):
    """Pre process the data co-variates.
    Categorical features will be one-hot-encoded.
    Numerical features will be standarized.
    Parameters
    ----------
    data: ndarray
    feature_names: list
    categories: dictionary
    Returns
    -------
    transformed data: ndarray
    """
    n_points = data.shape[0]
    processed_data = np.array([], dtype=np.float).reshape(n_points, 0)

    feature_names = (
        feature_names if feature_names else ["" for _ in range(data.shape[1])]
    )
    a = categories if categories else {}

    for idx, feature_name in enumerate(feature_names):
        if feature_name in categories:  # OneHotEncode
            enc = OneHotEncoder(categories="auto", sparse=False)
            data[np.argwhere(np.isnan(data[:, idx])), idx] = -1
            features = enc.fit_transform(data[:, idx].reshape(n_points, 1))
        else:  # Normalize
            features = data[:, idx] - np.mean(data[:, idx])
            features = features / np.std(data[:, idx])

        processed_data = np.concatenate(
            (processed_data, features.reshape(n_points, -1)), axis=1
        )
    return processed_data

def get_data_class(x, y, c):
    n = len(y)
    x_c = []
    for ii in range(n):
        if y[ii] == c:
            x_c.append(x[ii])
    x_c = np.array(x_c)
    return x_c, c * np.ones(x_c.shape[0], dtype=np.int64)

def subsample_data(x, y):
    seed=1
    generator = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    new_proportion = np.ceil(10 * np.min(counts)/100).astype(np.int64)
    c_maj = np.argmax(counts)
    x_maj, y_maj = get_data_class(x, y, c_maj)
    x_min, y_min = get_data_class(x, y, 1 - c_maj)
    subsampled_index = generator.choice(np.arange(x_maj.shape[0]), size=new_proportion, replace=False)
    x_maj = x_maj[subsampled_index]
    y_maj = y_maj[subsampled_index]
    new_x = np.concatenate([x_maj, x_min])
    new_y = np.concatenate([y_maj, y_min])

    shuffled_index = generator.permutation(np.arange(new_x.shape[0]))
    new_x = new_x[shuffled_index]
    new_y = new_y[shuffled_index]

    return new_x, new_y

def generate_dataset(name='adult', version=2, random_state=1):
    binarize = True

    dataset = fetch_openml(name, return_X_y=False, version=version, as_frame=False)

    data = dataset.data
    if type(data) is not np.ndarray:
        data = data.toarray()

    feature_names = dataset.feature_names
    categories = dataset.categories

    data = pre_process_data(data, feature_names, categories)
    target = dataset.target
    if binarize:  # One-hot encoding of target variable
        target = label_binarize(target, classes=np.unique(target)).ravel()
    # TODO: Think also about some preprocessing in case of regression
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.35, random_state=random_state)
    return x_train, y_train, x_test, y_test

def fetch_australian_dataset(seed):
    name='Australian'
    version=4
    x_tr, y_tr, x_te, y_te = generate_dataset(name=name, version=version, random_state=seed)
    return x_tr, y_tr, x_te, y_te
