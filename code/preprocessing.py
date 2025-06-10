import cv2

def preprocess_image(img, size=(256, 256)):
    """
    對輸入影像執行基本預處理：
    1. 調整大小
    2. 去雜訊（Gaussian Blur）
    3. 色彩空間轉換（RGB → HSV）

    參數:
        img (ndarray): 輸入影像 (BGR 格式)
        size (tuple): 輸出影像大小，預設為 (256, 256)

    回傳:
        預處理後的影像 (HSV 格式)
    """
    # Resize to fixed size
    resized = cv2.resize(img, size)

    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    # Convert to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return hsv