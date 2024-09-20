import numpy  as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def filter_out_highly_similar(df:pd.DataFrame, text_column:str,
                              similarity_threshold:float=0.95,
                              ngram:tuple=(1,3)) -> pd.DataFrame:
    """
    Maintain only documents whose similarity is below a threshold.
    Args:
        df (pd.core.frame.DataFrame): dataset;
        text_column (str): column name containing the texts;
        similarity_threshold (float): maximum similarity between pairs of documents to allow
        both to remain in dataset;
        ngram (tuple): tuple containing (min_tokens, max_tokens) in each feature of vectorized text.
    Return:
        dataset (pd.DataFrame): dataframe containing only documents whose text similarities are below threshold.
    """
    # If dataframe is empty, return itself
    if df.empty:
        return df

    # Assert text column is not empty
    assert text_column != '', "Text column is empty."

    # Assert text column is in dataframe
    assert text_column in df.columns, f"Text column {text_column} not in dataframe's columns."

    # Create output DataFrame
    df_out = df.copy(deep=True)
    df_out = df_out.reset_index(drop=True)
    
    # Vectorize the Text and Prepare Similarity Matrix
    vectorizer = TfidfVectorizer(ngram_range=ngram, dtype=np.float32)
    vectors = vectorizer.fit_transform(df_out[text_column])
    similarity_matrix = cosine_similarity(vectors)

    # Start Selecting the Files
    file_indexes = list(df_out.index)
    keep_indexes = [file_indexes[0]]
    for file_idx in file_indexes[1:]:
        try:
            if all(similarity_matrix[file_idx, out] < similarity_threshold
                   for out in keep_indexes):
                keep_indexes.append(file_idx)
        except Exception as error:
            print(f'[ERROR] Failed to get cosine similarity for {file_idx = } and {out = }.', error)

    # Filter elements to keep
    df_out = df_out.loc[keep_indexes].copy().reset_index(drop=True)
    
    # Return DataFrame
    return df_out
