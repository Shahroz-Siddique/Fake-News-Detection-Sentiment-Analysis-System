# src/models/train_ml.py
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_models(X_path, y_path):
    X = pickle.load(open(X_path, 'rb'))
    y = pickle.load(open(y_path, 'rb'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(),
        'MultinomialNB': MultinomialNB()
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            mlflow.log_param('model', name)
            mlflow.log_metric('accuracy', acc)
            mlflow.sklearn.log_model(model, f'model_{name}')
