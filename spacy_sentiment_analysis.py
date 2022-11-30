import string
import spacy
import pandas as pd
import re
from unidecode import unidecode

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

"""
@author: Sriya Madapusi Vasudevan
"""

def set_sentiment(ratingValue):
    """
    Preprocesses the data so that a sentiment is set based on rating value
    Rating Value --> Sentiment
    1, 2         --> 0 (Negative)
    3            --> 1 (Neutral)
    4, 5         --> 2 (Positive)
    Sentiment of -1 is set for any rating value that is not between 1-5
    :param ratingValue: (int) rating value (1-5) of a review
    :return: (int) sentiment value 0, 1 or 2
    """
    sentiment = -1
    if ratingValue == 1 or ratingValue == 2:
        sentiment = 0
    elif ratingValue == 3:
        sentiment = 1
    elif ratingValue == 4 or ratingValue == 5:
        sentiment = 2

    return sentiment


def read_data(filename, initial=False):
    """
    Reads data from given filename and returns a dataframe
    :param initial: (boolean) flag to check if its the first time reading a file and not
    :param filename: (string) name of the file
    :return: (DataFrame) dataframe
    """
    try:
        if initial:
            data_df = pd.read_csv(filename, delimiter='\t')
            data_df['Sentiment'] = data_df['RatingValue'].apply(lambda x: set_sentiment(x))

            # dropping all rows that don't have a valid sentiment
            data_df = data_df[data_df['Sentiment'] != -1]

            data_df = data_df[['Sentiment', 'Review']]
            return data_df
        else:
            data_df = pd.read_csv(filename)
            data_df.reset_index(drop=True, inplace=True)
            return data_df
    except FileNotFoundError:
        print("Error: File not found")


def balance_data(df):
    """
    Method balances data since there is more negative reviews than positive and neutral reviews
    :param df:(DataFrame) dataframe containing reviews and sentiment
    :return: (DataFrame) a balanced dataframe
    """
    random_state = 1

    num_pos = df.query('Sentiment == 2').shape[0]
    num_n = df.query('Sentiment == 1').shape[0]
    num_neg = df.query('Sentiment == 0').shape[0]

    print('---------------------------------')
    print('Before Balancing')
    print('---------------------------------')

    print(f"# positive reviews: {num_pos}\n# neutral reviews: {num_n}\n# negative reviews: {num_neg}")

    num_min_class = min([num_pos, num_n, num_neg])
    df1 = df.query('Sentiment == 2').sample(n=int(num_min_class))
    df2 = df.query('Sentiment == 1').sample(n=int(num_min_class))
    df3 = df.query('Sentiment == 0')

    bal_df = pd.concat([df1, df2, df3])
    bal_df = bal_df.sample(frac=1)
    bal_df = bal_df.reset_index(drop=True)

    num_pos = df1.shape[0]
    num_n = df2.shape[0]
    num_neg = df3.shape[0]

    print('---------------------------------')
    print('After Balancing')
    print('---------------------------------')

    print(f"# positive reviews: {num_pos}\n# neutral reviews: {num_n}\n# negative reviews: {num_neg}")
    return bal_df


def save_data(df):
    """
    Saves dataframe into two files: train and test csvs
    :param df: (DataFrame) a balanced dataframe
    :return: None
    """
    X = df['Review']
    y = df['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    train_df = pd.DataFrame()
    train_df['Sentiment'] = y_train
    train_df['Review'] = X_train
    train_df.to_csv("train.csv")

    test_df = pd.DataFrame()
    test_df['Sentiment'] = y_test
    test_df['Review'] = X_test
    test_df.to_csv("valid.csv")


def preprocessing(text, nlp):
    """
    Method to preprocess data - removes stop words, lemmatizes, removes punctuation, removes digits
    and only allowed english characters
    :param text: (String) a string that needs to be preocessed
    :param nlp: (Spacy) a spacy object loaded with only english dictionary
    :return: (String) a processed string
    """
    text = unidecode(text)
    text = text.lower()
    text = re.sub("\n", " ", text)

    doc = nlp(text)
    lemmatized_stopped_list = []

    for token in doc:
        if token.is_stop is False and len(token.lemma_) > 1:
            lemmatized_stopped_list.append(token.lemma_)

    tokenized_text = " ".join(lemmatized_stopped_list).strip()
    tokenized_text = tokenized_text.translate(str.maketrans('', '', string.punctuation))
    tokenized_text = tokenized_text.translate(str.maketrans('', '', string.digits))
    tokenized_text = ' '.join(tokenized_text.split())

    return tokenized_text


def fit_and_evaluate(classifier, X_train, y_train, X_test, y_test):
    """
    Fit and evaluate psased in model on training and test data
    :param classifier: (Sklearn Learner) A classification learner
    :param X_train: (Series) training set of predictors
    :param y_train: (Series) training set of outcome
    :param X_test: (Series) testing set of predictors
    :param y_test: (Series) testing set of outcome
    :return: (Series) predicted classes from X_test
    """
    text_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', classifier),
    ])

    text_clf.fit(X_train, y_train)
    y_pred = text_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"accuracy: {acc}")
    print(f"f1 score: {f1}")
    print("Confusion_matrix:")
    print(confusion_matrix(y_test, y_pred))

    return y_pred



# Uncomment if new data source and need to re-balance again
reviews_df = read_data('reviews.csv', initial=True)
balanced_df = balance_data(reviews_df)
save_data(balanced_df)


# read in the train and test set
train_df = read_data('train.csv')
test_df = read_data('valid.csv')

print("---------------------------------")
print(f"Training set: {train_df.shape}")
print(f"Training set: {test_df.shape}")

nlp = spacy.load("en_core_web_sm")

X_train = train_df['Review'].apply(lambda x: preprocessing(x, nlp))
y_train = train_df['Sentiment']
X_test = test_df['Review'].apply(lambda x: preprocessing(x, nlp))
# X_test = test_df['Review']
y_test = test_df['Sentiment']

# X_train.to_csv('prepr_train.csv')
# X_test.to_csv('prepr_test.csv')

"""
print("---------------------------------")
print("Naive Bayes")
print("---------------------------------")

mnb = MultinomialNB()
fit_and_evaluate(mnb, X_train, y_train, X_test, y_test)
"""

print("---------------------------------")
print("SVM")
print("---------------------------------")

svm = SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)
y_predictions = fit_and_evaluate(svm, X_train, y_train, X_test, y_test)

print("---------------------------------")
print("On average, SVM performed better than the rest")
print("---------------------------------")

"""
print("---------------------------------")
print("Random Forest")
print("---------------------------------")
rf_clf = RandomForestClassifier()
fit_and_evaluate(rf_clf, X_train, y_train, X_test, y_test)

print("---------------------------------")
print("Gradient Boosting")
print("---------------------------------")
gb_clf = GradientBoostingClassifier()
fit_and_evaluate(gb_clf, X_train, y_train, X_test, y_test)

print("---------------------------------")
print("Decision Tree")
print("---------------------------------")
bagging = BaggingClassifier(DecisionTreeClassifier())
fit_and_evaluate(bagging, X_train, y_train, X_test, y_test)
"""