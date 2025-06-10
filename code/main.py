import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

# 匯入模組（你需分別實作這些）
from preprocessing import preprocess_image
from segmentation import segment_image
from feature_extraction import extract_features

# ===== 可調參數區 =====
DATA_DIR = "./Dataset/Classification_dataset"
FEATURE_TYPE = "color_texture"  # 可選: 'color', 'texture', 'shape', 'color_texture' 等
MODEL_PARAMS = {"n_neighbors": 5, "weights": "distance"}
TEST_RATIO = 0.2  # 訓練測試比例（0.2 表示 80% 訓練 20% 測試）
# =====================

def load_data(data_dir):
    X = []  # 特徵
    y = []  # 標籤
    class_names = sorted(os.listdir(data_dir))
    for label, class_folder in enumerate(class_names):
        class_path = os.path.join(data_dir, class_folder)
        for fname in os.listdir(class_path):
            img_path = os.path.join(class_path, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            preprocessed = preprocess_image(img)
            segmented = segment_image(preprocessed)            
            features = extract_features(segmented, mode=FEATURE_TYPE)
            
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y), class_names

def main():
    print("[INFO] Loading and processing data...")
    X, y, class_names = load_data(DATA_DIR)

    print("[INFO] Splitting data (train/test ratio: {:.2f})...".format(1 - TEST_RATIO))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO, stratify=y, random_state=42)

    print("[INFO] Scaling feature vectors...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("[INFO] Training KNN model...")
    model = KNeighborsClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()