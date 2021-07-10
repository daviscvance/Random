from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
import pickle
import ast
import sys
import os

pd.options.display.max_rows = 50
pd.options.display.max_columns = 20
pd.options.display.max_colwidth = 100

def read_quiz_questions() -> pd.DataFrame:
    """Read Newsela data and convert to Pandas DataFrame

    Returns:
        df::pd.DataFrame
            Object holding quiz text and scores for processing.
    """
    data_path = './data/quiz_question_data.txt'
    if not os.path.exists(data_path):
        raise Exception(f'Data not found in {data_path}')

    with open(data_path, 'r') as f:
        questions = f.read()

    return pd.DataFrame(ast.literal_eval(questions))

def define_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Defines a success level for classification."""

    df['success_lvl'] = np.select(
        [
            # There seems to be rounding errors in the data, so using a wider
            # range than necessary to capture them.
            df.percent_correct.between(-1.0, 0.58, inclusive='right'),
            df.percent_correct.between(0.58, 2.00, inclusive='right'),
            # df.percent_correct.between(.66, 1., inclusive='right')
        ],
        [
            'low',
            # 'medium',
            'high'
        ],
        default=None
    )
    return df

def vectorize_text(df: pd.DataFrame):
    """Vectorizes DataFrame with count and tf-idf matrices.

    Count vectorization and term frequency-inverse document frequency (tf-idf)
    vectorization will provide term measurement for documents for use in
    analysis. Final transformed data is pickled during this process.

    Args:
        df::pd.DataFrame Text & score data.

    Returns:
        df_cv::pd.DataFrame
            Processed data with sparse columns added for count vectorized text.
        ct_vec::scipy.sparse.csr.csr_matrix
            Count Vectorized sparse result matrix.
        ct_names::list
            List of the ct_vec feature names mapped for matrix.
        df_tfidf::pd.DataFrame
            Processed data with sparse columns added for tfidf vectorized text.
        tfidf::scipy.sparse.csr.csr_matrix
            Tf-Idf Vectorized sparse result matrix.
        tf_names::list
            List of the tf-idf feature names mapped for matrix. Matches
            (or should match) ct_names.
    """
    # Creating a stop_words list set that are common to many questions.
    common_phrases = [
        'read the sentence from the passage',
        'which of the following best describes',
        'which is the best one sentence * for the section',
        'which sentence from the passage provides the most evidence'
        'select the sentence that does not support the central idea of the article',
        'supports the main idea',
        'select the paragraph from the section that explains how that shows the ',
        'that is most relevant to be included in the summary of the article',
        'according to the article',
        'which of these is not one',
    ]
    stop_words = stopwords.words('english')
    [stop_words.extend(x.split()) for x in common_phrases]

    ct_vectorizer = CountVectorizer(token_pattern='\\w{3,}',
                                    max_df=.3,
                                    min_df=.001,
                                    stop_words=list(set(stop_words)),
                                    strip_accents='ascii',  # Faster than unicode.
                                    ngram_range=(1, 3),  # Enable uni, bi, trigrams.
                                    lowercase=True,
                                    dtype='uint8')

    tfidf_vectorizer = TfidfVectorizer(token_pattern='\\w{3,}',
                                       max_df=.3,
                                       min_df=.001,
                                       stop_words=list(set(stop_words)),
                                       strip_accents='ascii',  # Faster than unicode.
                                       ngram_range=(1, 3),  # Enable uni, bi, trigrams.
                                       lowercase=True,
                                       sublinear_tf=True,  # Replace tf with 1 + log(tf).
                                       smooth_idf=True,  # Default 1 doc for each term.
                                       dtype=np.float32)

    # Count & tf-idf vectorization learns vocab and transforms data into matrices.
    ct_vec = ct_vectorizer.fit_transform(np.array(df.text))
    tfidf = tfidf_vectorizer.fit_transform(np.array(df.text))
    # print("Shape of ct_vec:", ct_vec.shape)
    # print('Size of ct_vec:', sys.getsizeof(ct_vec))
    # print("Shape of tfidf:", tfidf.shape)
    # print('Size of tfidf:', sys.getsizeof(tfidf), '\n')

    ct_names = ct_vectorizer.get_feature_names()
    tf_names = tfidf_vectorizer.get_feature_names()

    df_cv = pd.concat(
        [df, pd.DataFrame(ct_vec.toarray(), columns=ct_names)],
        axis=1)
    df_tfidf = pd.concat(
        [df, pd.DataFrame(tfidf.toarray(), columns=tf_names)],
        axis=1)

    return (
        df_cv,
        ct_vec,
        ct_names,
        df_tfidf,
        tfidf,
        tf_names
    )

def naive_bayes_classify(df: pd.DataFrame, vect, names):
    """Classifies success data with a naive bayes model

    Args:
        df::pd.DataFrame Feature set and target, used for just target.
        vect::scipy.sparse.csr.csr_matrix
            Vectorized text data as the feature set to train classifier.
        names::List
            List of feature names to map the results and provide detail on
            important words or phrases.
    """
    features = vect
    target = df.success_lvl

    X_train, X_test, y_train, y_test = \
        train_test_split(features, target, test_size=0.2, random_state=42)

    nb_clf = MultinomialNB()
    nb_clf.fit(X_train, y_train)
    nb_predictions = nb_clf.predict(X_test)
    print('Accuracy score for Naive Bayes:', accuracy_score(y_test, nb_predictions))


    # Find Top/Bottom num of terms used to describe the classes.
    num = 10
    low_class_prob_sorted = nb_clf.feature_log_prob_[0, :].argsort()[::-1]
    hi_class_prob_sorted = nb_clf.feature_log_prob_[1, :].argsort()[::-1]
    print('\n', f'Low score Top{num} phrases:', np.take(names, low_class_prob_sorted[:num]))
    print('\n', f'Low score Bot{num} phrases:', np.take(names, low_class_prob_sorted[-num:]))
    print('\n', f'High score Top{num} phrases:', np.take(names, hi_class_prob_sorted[:num]))
    print('\n', f'High score Bot{num} phrases:', np.take(names, hi_class_prob_sorted[-num:]))

def extract_frequent_words(df:pd.DataFrame):
    """Prints rank sorted pivot to find the most frequent words.

    Explains high scoring ranks and low ranking scores.

    Args:
        df::pd.DataFrame
            The vectorized text in a DataFrame.
    """
    x = (pd.pivot_table(df.drop(['text', 'percent_correct'], axis=1),
                       index='success_lvl',
                       aggfunc=['sum', 'mean'])  # Count shows ~50/50 split
           .transpose()
           .loc[:, ['high', 'low']]
           .unstack(level=0))

    # Rank the most frequent phrases
    x['high_rank'] = x[('high', 'sum')].rank(method='dense', ascending=False)
    x['low_rank'] = x[('low', 'sum')].rank(method='dense', ascending=False)
    print(x[x.high_rank <= 10.].sort_values('high_rank'))
    print(x[x.low_rank <= 10.].sort_values('low_rank'))

if __name__ == '__main__':
    # df_cv, df_tfidf = vectorize_text(define_categorical(read_quiz_questions()))
    df_cv, ct_vec, ct_names, df_tfidf, tfidf, tf_names = \
        vectorize_text(define_categorical(read_quiz_questions()))

    # Tf-Idf doesnt provide any extra value, so its excluded below.
    print('\n', '='*10, 'Count Vectorization', '='*70)
    naive_bayes_classify(df_cv, ct_vec, ct_names)
    # print('\n', '='*10, 'Tf-Idf', '='*83)
    # naive_bayes_classify(df_tfidf, tfidf, tf_names)

    print('\n', '='*10, 'Count Vectorization', '='*70)
    extract_frequent_words(df_cv)
    # print('\n', '='*10, 'Tf-Idf', '='*83)
    # extract_frequent_words(df_tfidf)
