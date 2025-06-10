import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops

def extract_color_hist(img_bgr, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_texture_features(img_bgr, distances=[1], angles=[0], props=['contrast', 'homogeneity']):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(gray / 4)  # 壓縮到 64 等級 (0~255 → 0~63)
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)
    features = []
    for prop in props:
        features.append(graycoprops(glcm, prop).flatten())
    return np.concatenate(features)

def extract_shape_features(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments)
    return hu.flatten()

def extract_features(img_bgr, mode='color_texture'):
    features = []

    if mode in ['color', 'color_texture']:
        color_feat = extract_color_hist(img_bgr)
        if color_feat is not None and len(color_feat) > 0:
            features.extend(color_feat)
        else:
            print("[WARN] color feature is empty")

    if mode in ['texture', 'color_texture']:
        texture_feat = extract_texture_features(img_bgr)
        if texture_feat is not None and len(texture_feat) > 0:
            features.extend(texture_feat)
        else:
            print("[WARN] texture feature is empty")

    if mode == 'shape':
        shape_feat = extract_shape_features(img_bgr)
        if shape_feat is not None and len(shape_feat) > 0:
            features.extend(shape_feat)
        else:
            print("[WARN] shape feature is empty")

    if len(features) == 0:
        print("[ERROR] All features are empty for this image")
        return None

    return np.array(features, dtype=np.float32)
