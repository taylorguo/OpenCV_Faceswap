# OpenCV Faceswap 

Using OpenCV3.x to faceswap

使用OpenCV3.0 对图像人脸进行变脸

## 依赖安装
python3.6

```
pip install -r requirements.txt
```

[Download dlib landmark model](https://pan.baidu.com/s/1Bbn7Zl57do969b76ui0OeA)

## 使用方法

```
python faceswap_cv.py ./path_to/人脸素材图片.jpg ./path_to/被替换的人脸素材图片.jpg
```

举个例子：

```
python faceswap_cv.py ./samples/fbb_001.jpeg ./samples/trump_smile.jpg
```

## 结果展示
![](samples/sample.jpg)