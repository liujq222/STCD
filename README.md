# Side-scan Sonar Image Preprocessing
侧扫声呐图像预处理，基于 OpenCV + Pillow 实现，针对 SCTD 声呐数据集，完成图像灰度转换、去噪、对比度增强与可视化，适用于水下目标检测前置处理。

## ✨ 项目特性
- 支持带中文路径的声呐图像读取，兼容 Pillow 读取，规避 OpenCV 中文路径报错
- 图像处理流水线：原图 → 灰度化 → 高斯滤波 + 中值联合去噪 → CLAHE 自适应直方图均衡
- 输出多子图对比可视化结果，直观展示每一步预处理效果，可直接放入实验报告
- 附带灰度直方图绘制，量化观察灰度分布变化，分析对比度增强效果
- 可扩展：增加全局直方图均衡作为对照，对比全局/自适应均衡的优缺点

## 📁 目录结构
sonar-preprocess/
├── sonar_process.py # 主处理脚本
├── img/ # 存放侧扫声呐原图（SCTD 数据集）
├── output/ # 输出处理后的对比图、直方图结果
└── README.md

## 🛠️ 环境依赖
```bash
pip install opencv-python pillow matplotlib numpy
