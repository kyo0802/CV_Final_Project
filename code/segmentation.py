import cv2
import numpy as np

def segment_image(img_hsv):
    """
    使用 Otsu 閾值 + 形態學運算來從 HSV 影像中分割主體區域。
    處理流程：
    1. 取 HSV 中 V（亮度）通道
    2. 用 Otsu 的方法進行二值化
    3. 形態學開運算去除雜訊
    4. 產生分割 mask，並套用在原始影像上

    參數:
        img_hsv (ndarray): 輸入影像（HSV 格式）

    回傳:
        result_bgr (ndarray): 分割後保留主體區域的 BGR 影像
    """
    # 取得 S 和 V 通道
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]

    # 使用 Otsu 找出自動門檻
    s_thresh, _ = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thresh, _ = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 調整閾值
    s_thresh = max(0, s_thresh - 10)
    v_thresh = min(255, v_thresh + 10)

    # 二值化處理
    _, s_binary = cv2.threshold(s_channel, s_thresh, 255, cv2.THRESH_BINARY)
    _, v_binary = cv2.threshold(v_channel, v_thresh, 255, cv2.THRESH_BINARY_INV)

    # 合併條件產生 mask
    mask = np.logical_and(s_binary > 0, v_binary > 0).astype(np.uint8) * 255

    # 形態學清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 只保留最大連通區域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    
    # 將主體從 HSV 抽出，並轉換回 BGR
    segmented_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    result_bgr = cv2.cvtColor(segmented_hsv, cv2.COLOR_HSV2BGR)

    return result_bgr