import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords

# Extract

def import_data(folder_path : str):
    """
    Import the source files from the local folder
    """

    ds_names = pd.read_csv(f"{folder_path}/IMDb movies.csv")
    ds_ratings = pd.read_csv(f"{folder_path}/IMDb ratings.csv")

    return ds_names, ds_ratings

# Transform

def filter_names(ds : pd.DataFrame) -> pd.DataFrame:
    """
    The purpose of this function is to return a filtered list of names
    """
    _ds = ds.copy()
    _ds = _ds[_ds['year'] != 1980]

    return _ds

def filter_ratings(ds : pd.DataFrame) -> pd.DataFrame:
    """
    The purpose of this function is to return a filtered list of ratings
    """
    _ds = ds.copy()

    _ds = _ds[_ds['total_votes'] > 200]

    return _ds

def join_datasets(ds_names : pd.DataFrame, ds_ratings : pd.DataFrame) -> pd.DataFrame:
    """
    Take in two datasets and return a single joined dataset
    """
    _ds = ds_names.merge(ds_ratings, on='imdb_title_id')

    return _ds


def generate_bag_of_words(ds : pd.DataFrame):
    """[summary]

    Args:
        ds (pd.DataFrame): Feed in the dataframe that contains the 'title' column.

    Returns:
        [word_counts, text_processing]: [Returns both the dataframe with the word counts and the model]
    """
    text_processing = Pipeline([('vect', CountVectorizer(stop_words=list(set(stopwords.words('english')))))])

    text_processing_output = text_processing.fit_transform(ds['title'])

    word_counts = pd.DataFrame(text_processing_output.toarray(),
                      columns=text_processing['vect'].get_feature_names())

    # Including only the top 20 words
    _top_word_columns = word_counts.sum().T.sort_values(ascending=False).head(20).index

    word_counts = word_counts[_top_word_columns]


    return word_counts, text_processing

#Load

def save_parquet(ds : pd.DataFrame, filename : str) -> None:
    """
    Saves the dataframe as a parquet
    """

    ds.to_parquet(f"{filename}.parquet")
