import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_features(df, text_column, save_path_X, save_path_y, save_vectorizer_path):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df[text_column])
    y = df['label']
    with open(save_path_X, 'wb') as fx, open(save_path_y, 'wb') as fy, open(save_vectorizer_path, 'wb') as fv:
        pickle.dump(X, fx)
        pickle.dump(y, fy)
        pickle.dump(vectorizer, fv)
    return X, y, vectorizer
