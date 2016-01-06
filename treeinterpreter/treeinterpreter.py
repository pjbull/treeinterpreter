# -*- coding: utf-8 -*-
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from distutils.version import LooseVersion
import sklearn

if LooseVersion(sklearn.__version__) < LooseVersion("0.17"):
    raise Exception("treeinterpreter requires scikit-learn 0.17 or later")

try:
    import joblib
    from joblib import Parallel, delayed

    if LooseVersion(joblib.__version__) < LooseVersion("0.9.3"):
        raise Exception("treeinterpreter requires joblib 0.9.3 or later")

    SUPPORTS_PARALLEL = True
except ImportError:
    SUPPORTS_PARALLEL = False


def _get_tree_paths(tree, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    if left_child != _tree.TREE_LEAF:
        left_paths = _get_tree_paths(tree, left_child, depth=depth + 1)
        right_paths = _get_tree_paths(tree, right_child, depth=depth + 1)

        for path in left_paths:
            path.append(node_id)
        for path in right_paths:
            path.append(node_id)
        paths = left_paths + right_paths
    else:
        paths = [[node_id]]
    return paths


def _process_leaf(feature_mat, row, leaf, paths, values, line_shape):
    """
    Processes a single leaf and returns the row number, bias, and
    contributions.
    """
    for path in paths:
        if leaf == path[-1]:
            break

    bias = values[path[0]]
    contribs = np.zeros(line_shape)
    for i in range(len(path) - 1):
        contrib = values[path[i+1]] - values[path[i]]
        contribs[feature_mat[path[i]]] += contrib

    contribution = contribs
    return row, bias, contribution

def _predict_tree(model, X):
    """
    For a given DecisionTreeRegressor or DecisionTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    leaves = model.apply(X)
    paths = _get_tree_paths(model.tree_, 0)

    for path in paths:
        path.reverse()

    # remove the single-dimensional inner arrays
    values = model.tree_.value.squeeze()
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])

    if type(model) == DecisionTreeRegressor:
        contributions = np.zeros(X.shape)
        biases = np.zeros(X.shape[0])
        line_shape = X.shape[1]
    elif type(model) == DecisionTreeClassifier:
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.zeros((X.shape[0], model.n_classes_))
        contributions = np.zeros((X.shape[0],
                                  X.shape[1], model.n_classes_))
        line_shape = (X.shape[1], model.n_classes_)

    for row, leaf in enumerate(leaves):
        for path in paths:
            if leaf == path[-1]:
                break
        biases[row] = values[path[0]]
        contribs = np.zeros(line_shape)
        for i in range(len(path) - 1):
            contrib = values[path[i+1]] - \
                      values[path[i]]
            contribs[model.tree_.feature[path[i]]] += contrib
        contributions[row] = contribs

    direct_prediction = values[leaves]

    return direct_prediction, biases, contributions


def _predict_tree_parallel(X, paths, tree_vals, leaves, model_type, n_classes_, tree_features):
    """
    For a given DecisionTreeRegressor or DecisionTreeClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    for path in paths:
        path.reverse()

    # remove the single-dimensional inner arrays
    values = tree_vals.squeeze()
    # reshape if squeezed into a single float
    if len(values.shape) == 0:
        values = np.array([values])

    if model_type == DecisionTreeRegressor:
        contributions = np.zeros(X.shape)
        biases = np.zeros(X.shape[0])
        line_shape = X.shape[1]
    elif model_type == DecisionTreeClassifier:
        # scikit stores category counts, we turn them into probabilities
        normalizer = values.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        values /= normalizer

        biases = np.zeros((X.shape[0], n_classes_))
        contributions = np.zeros((X.shape[0],
                                  X.shape[1], n_classes_))
        line_shape = (X.shape[1], n_classes_)

    for row, leaf in enumerate(leaves):
        for path in paths:
            if leaf == path[-1]:
                break
        biases[row] = values[path[0]]
        contribs = np.zeros(line_shape)
        for i in range(len(path) - 1):
            contrib = values[path[i+1]] - \
                      values[path[i]]
            contribs[tree_features[path[i]]] += contrib
        contributions[row] = contribs

    direct_prediction = values[leaves]

    return direct_prediction, biases, contributions


def _predict_forest(model, X, n_jobs, verbose, batch_size):
    """
    For a given RandomForestRegressor or RandomForestClassifier,
    returns a triple of [prediction, bias and feature_contributions], such
    that prediction ≈ bias + feature_contributions.
    """
    biases = []
    contributions = []
    predictions = []

    if n_jobs == 1:
        for tree in model.estimators_:
            pred, bias, contribution = _predict_tree(tree, X)
            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)
    else:
        tasks = [delayed(_predict_tree_parallel)(X,
                                                 _get_tree_paths(tree.tree_, 0),
                                                 tree.tree_.value,
                                                 tree.apply(X),
                                                 type(tree),
                                                 tree.n_classes_,
                                                 tree.tree_.feature)
                    for tree in model.estimators_]
        results = Parallel(n_jobs=n_jobs, verbose=verbose, batch_size=batch_size)(tasks)
        predictions, biases, contributions = zip(*results)

    return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
            np.mean(contributions, axis=0))


def predict(model, X, n_jobs=1, verbose=0, batch_size='auto'):
    """ Returns a triple (prediction, bias, feature_contributions), such
    that prediction ≈ bias + feature_contributions.
    Parameters
    ----------
    model : DecisionTreeRegressor, DecisionTreeClassifier or
        RandomForestRegressor, RandomForestClassifier
    Scikit-learn model on which the prediction should be decomposed.

    X : array-like, shape = (n_samples, n_features)
    Test samples.

    n_jobs [optional]: Use joblib to parallelize the computation with n_jobs
        processes; default is 1 job (serial). Note: only forests will be
        run in parallel.

    verbose [optional]: Print debug information from joblib by increasing
        this setting; default is 0.

    batch_size [optional]: The number of trees to pass at a time to joblib; the
        default ('auto') should work well in most cases.

    Returns
    -------
    decomposed prediction : triple of
    * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
        for classification
    * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
        classification
    * contributions, shape = (n_samples, n_features) for regression or
        shape = (n_samples, n_features, n_classes) for classification
    """
    # Only single out response variable supported,
    if model.n_outputs_ > 1:
        raise ValueError("Multilabel classification trees not supported")

    if n_jobs > 1 and not SUPPORTS_PARALLEL:
        raise ValueError("If joblib is not installed, treeinterpreter only supports n_jobs=1")

    if (type(model) == DecisionTreeRegressor or
        type(model) == DecisionTreeClassifier):
        return _predict_tree(model, X)
    elif (type(model) == RandomForestRegressor or
          type(model) == RandomForestClassifier):
        return _predict_forest(model, X, n_jobs=n_jobs, verbose=verbose, batch_size=batch_size)
    else:
        raise ValueError("Wrong model type. Base learner needs to be \
            DecisionTreeClassifier or DecisionTreeRegressor.")

if __name__ == "__main__":
    # test
    from sklearn.datasets import load_iris
    iris = load_iris()
    idx = range(len(iris.data))
    np.random.shuffle(idx)
    X = iris.data[idx]
    Y = iris.target[idx]
    dt = RandomForestClassifier(max_depth=20, n_estimators=1500)
    dt.fit(X[:len(X)/2], Y[:len(X)/2])
    testX = X[len(X)/2:len(X)/2+5]
    base_prediction = dt.predict_proba(testX)
    pred, bias, contrib = predict(dt, testX, n_jobs=2, verbose=10)

    assert(np.allclose(base_prediction, pred))
    assert(np.allclose(pred, bias + np.sum(contrib, axis=1)))
