# tests/test_model.py
import pickle
from sklearn.metrics import accuracy_score

def test_model_accuracy(model_path, X_test, y_test):
    model = pickle.load(open(model_path, 'rb'))
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    assert acc > 0.80, f"Expected accuracy > 0.80, but got {acc}"
