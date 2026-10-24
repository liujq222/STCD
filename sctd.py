from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ===================== 修改这里为你的图片路径 =====================
img_path = r"D:\桌面\classwork\sctd\JPEGImages\000001.jpg"
# =================================================================

# 1. PIL读取（兼容中文路径，解决cv2.imread中文路径bug）
try:
    pil_img = Image.open(img_path)
    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    print("✅ 图片读取成功")
except Exception as e:
    print("❌读取失败：", e)
    exit()

# 2. 灰度化
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 3. 滤波：高斯模糊 + 中值滤波（侧扫声呐图去除椒盐噪声、条纹噪声）
gauss_blur = cv2.GaussianBlur(gray, (3, 3), sigmaX=0)
median_blur = cv2.medianBlur(gauss_blur, ksize=3)

# 4. 直方图均衡化（CLAHE，限制对比度自适应均衡，声呐优先用这个，避免过度增强噪声）
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(median_blur)

# 计算灰度直方图
hist_gray = cv2.calcHist([gray], channels=[0], mask=None, histSize=[256], ranges=[0, 256])
hist_clahe = cv2.calcHist([img_clahe], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# 5. 绘图展示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文
plt.figure(figsize=(14, 7))

plt.subplot(2, 5, 1)
plt.title("原始声呐图")
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(2, 5, 2)
plt.title("灰度图")
plt.imshow(gray, cmap="gray")
plt.axis("off")

plt.subplot(2, 5, 3)
plt.title("高斯+中值滤波去噪")
plt.imshow(median_blur, cmap="gray")
plt.axis("off")

plt.subplot(2, 5, 4)
plt.title("CLAHE均衡增强")
plt.imshow(img_clahe, cmap="gray")
plt.axis("off")

plt.subplot(2, 5, 5)
plt.title("原图灰度直方图")
plt.plot(hist_gray, color="k")
plt.xlim([0, 256])

plt.subplot(2, 5, 10)
plt.title("均衡后直方图")
plt.plot(hist_clahe, color="k")
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

# 6. 保存结果
cv2.imwrite("sctd_gray.jpg", gray)
cv2.imwrite("sctd_filter.jpg", median_blur)
cv2.imwrite("sctd_clahe.jpg", img_clahe)
print("✅处理图片已保存到代码同目录")