from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_knn(X_train, y_train, model_params=None):
    """
    訓練 KNN 模型
    """
    if model_params is None:
        model_params = {"n_neighbors": 3, "weights": "uniform"}
    model = KNeighborsClassifier(**model_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    評估模型準確率與分類報告
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Accuracy: {acc * 100:.2f}%")
    print("[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
