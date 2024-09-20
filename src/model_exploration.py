
import time
import warnings
import numpy  as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import Tuple
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from src.sklearn_algorithms import AlgorithmModels
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (precision_score, f1_score, make_scorer,
                             recall_score, confusion_matrix)


def get_embeddings(X_train_text:list, X_test_text:list, random_state:int = 91,
                   ngram_range:Tuple = (1,1), min_df:float = 0.05,
                   verbose:bool = False) -> Tuple[int, list, list]:
    """
    Generate TF-IDF embeddings for a given list of training and test text data.

    Parameters:
    - X_train_text (list): List of training text data.
    - X_test_text (list): List of test text data.
    - random_state (int): Random state for reproducibility. Defaults to 91.
    - ngram_range (tuple): N-gram range for TF-IDF vectorization. Defaults to (1,1).
    - min_df (float): Minimum document frequency for TF-IDF vectorization. Defaults to 0.05.
    - verbose (bool): Whether to print additional information. Defaults to False.

    Returns:
    - Tuple[int, list, list]: A tuple containing the number of components used, the training embeddings,
    and the test embeddings.
    """
    tfidf_model = TfidfVectorizer(dtype=np.float32, ngram_range = ngram_range,
                                  min_df = min_df).fit(X_train_text)
    tfidf_train = tfidf_model.transform(X_train_text)
    tfidf_test  = tfidf_model.transform(X_test_text)

    # Dimensionality Reduction (only over train data)
    tsvd      = TruncatedSVD(n_components=min(1000, int(np.floor(0.75*tfidf_train.shape[1]))),
                             n_iter=10, random_state=random_state)
    vct       = tsvd.fit(tfidf_train)
    variance  = np.cumsum(vct.explained_variance_ratio_) # explained_variance
    size      = len(variance[variance < 0.75]) # cumulative explained variance ratio is 75%

    if verbose:
        print(f'[INFO] TF-IDF {ngram_range} embeddings with final size = {size}')

    svd_model = TruncatedSVD(n_components=size, n_iter=10,
                             random_state=random_state).fit(tfidf_train)
    return ( size, list(svd_model.transform(tfidf_train)), list(svd_model.transform(tfidf_test)) )


def explore_crossval(X_train:list, y_train:list,
                     X_test:list, y_test:list,
                     random_state:int = 91, cv:int=3) -> pd.DataFrame:
    """
    Evaluate the performance of different classification algorithms using cross-validation.

    Parameters:
    - X_train: List of training data features.
    - y_train: List of training data labels.
    - X_test: List of test data features.
    - y_test: List of test data labels.
    - random_state: Integer value for reproducibility. Default is 91.
    - cv: Integer value for the number of folds in cross-validation. Default is 3.

    Returns:
    - pandas.DataFrame: A DataFrame containing the evaluation metrics for each algorithm.
    """
    # Evaluate Unique Classes
    unique_class_label = set(y_train + y_test)
    if len(unique_class_label) < 2:
        raise Exception(f'[ERROR] Single or none available label = {unique_class_label}')

    # Load Models to Explore
    model_list = AlgorithmModels(len(unique_class_label), random_state=random_state)
    if len(unique_class_label) == 2:  # 2 Classes = Binary
        average = 'binary'
        classification_algorithms = model_list.get_all_binary_models()
    else:  # More than 2 Classes = Multiclass
        average = 'weighted'
        classification_algorithms = model_list.get_all_multiclass_models()

    row_metrics = list()
    for model_name, model in tqdm(classification_algorithms.items()):
        # Prepare the metrics
        metrics = dict()
        metrics['model'] = model_name
        metrics['average'] = average

        # Train the Model
        start = time.time()
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            warnings.simplefilter('ignore', category=FutureWarning)
            trained_model = deepcopy(model)
            trained_model.fit(X_train, y_train)
        metrics['training_time(s)'] = "{:.3f}".format(time.time() - start)

        # Get Predictions
        start = time.time()
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore', category=FutureWarning)
            y_pred = trained_model.predict(X_test)
        metrics['prediction_time(s)'] = "{:.3f}".format(time.time() - start)

        # Generate metrics
        metrics['precision'] = precision_score(y_test, y_pred, average=average, zero_division=0)
        metrics['recall']    = recall_score(y_test, y_pred, average=average, zero_division=0)
        metrics['F1']        = f1_score(y_test, y_pred, average=average, zero_division=0)

        # Add Misclassifications
        conf_matrix = confusion_matrix(y_test, y_pred)
        diag_conf_matrix = np.diag(conf_matrix).tolist()
        metrics['misclassification'] = f'{conf_matrix.sum() - np.sum(diag_conf_matrix)} (out of {conf_matrix.sum()})'

        # For Cross-Validation, we categorize the Labels to INT values
        labels_categories = dict()
        y_train_cat = list()
        for elem in y_train:
            if elem not in labels_categories.keys():
                labels_categories[elem] = len(labels_categories)
            y_train_cat.append(labels_categories[elem])

        # Add Crossvalidation Scores
        scoring = {'crossval_precision': make_scorer(precision_score, average=average, zero_division=0),
                   'crossval_recall':    make_scorer(recall_score, average=average, zero_division=0),
                   'crossval_F1':        make_scorer(f1_score, average=average, zero_division=0), }
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            warnings.simplefilter('ignore', category=FutureWarning)
            model_crossval = deepcopy(model)
            scores = cross_validate(model_crossval, X_train, y_train_cat,
                                    scoring=scoring, cv=cv)
        metrics['crossval_precision(avg)'] = scores['test_crossval_precision'].mean()
        metrics['crossval_precision(std)'] = scores['test_crossval_precision'].std()
        metrics['crossval_recall(avg)']    = scores['test_crossval_recall'].mean()
        metrics['crossval_recall(std)']    = scores['test_crossval_recall'].std()
        metrics['crossval_F1(avg)']        = scores['test_crossval_F1'].mean()
        metrics['crossval_F1(std)']        = scores['test_crossval_F1'].std()

        # Add Metrics in DataFrame
        row_metrics.append(metrics)

    # Return Metrics DataFrame
    return pd.DataFrame(row_metrics)


def make_confusion_matrix(y_test:list, y_pred:list,
                          title:str, output_path:str = './out',
                          cbar:bool=True, figsize=(10,10),
                          cmap:str='Blues') -> None:
    """
    Creates a confusion matrix from the true labels (y_test) and predicted labels (y_pred).

    Args:
        y_test (list): A list of true labels.
        y_pred (list): A list of predicted labels.
        title (str): The title of the confusion matrix.
        output_path (str, optional): The path to save the confusion matrix image. Defaults to './out'.
        cbar (bool, optional): Whether to include a color bar in the confusion matrix. Defaults to True.
        figsize (tuple, optional): The size of the confusion matrix figure. Defaults to (10,10).
        cmap (str, optional): The color map to use in the confusion matrix. Defaults to 'Blues'.
    """
    # Prepare confusion matrix
    categories = sorted(list(set(y_test) | set(y_pred)))
    cf      = confusion_matrix(y_test, y_pred, labels=categories)
    cf_prob = confusion_matrix(y_test, y_pred, labels=categories, normalize='true')

    # Prepare Groups Counts
    group_counts      = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_prob.flatten()]

    box_labels = [f"{v1}{v2}".strip() for v1, v2 in zip(group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # Heatmap Visualization
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('white')
    sns.heatmap(cf_prob*100,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,
                xticklabels=categories,yticklabels=categories)

    # Labels
    plt.title(title)
    plt.ylabel('Annotation')
    plt.xlabel('Prediction')

    # Show
    plt.tight_layout()
    chart_file_path = f'{output_path}/confusion_matrix_{title.replace(" ", "_")}.png'
    if output_path:
        plt.savefig(chart_file_path)
    plt.show()
    plt.close()
