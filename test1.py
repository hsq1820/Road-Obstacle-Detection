#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import h5py
import numpy as np
import cv2
import tqdm as tqdm
from models import build_stixel_net
from tensorflow.keras.models import load_model
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
import tensorflow.keras.backend as K

# 禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 固定模型路径
MODEL_PATH = "D:/Road Obstacle Detection/saved_models/model-033.h5"


def test_single_image(model, img, label_size=(100, 50)):
    assert img is not None

    h, w, c = img.shape
    val_aug = Compose([Resize(370, 800), Normalize(p=1.0)])
    aug_img = val_aug(image=img)["image"]
    aug_img = aug_img[np.newaxis, :]
    predict = model.predict(aug_img, batch_size=1)
    predict = K.reshape(predict, label_size)
    predict = K.eval(K.argmax(predict, axis=-1))

    for x, py in enumerate(predict):
        x0 = int(x * w / 100)
        x1 = int((x + 1) * w / 100)
        y = int((py + 0.5) * h / 50)
        cv2.rectangle(img, (x0, 0), (x1, y), (0, 0, 255), 1)

    return img


def check_h5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件 {file_path} 是有效的HDF5文件")
            print("文件结构:")
            f.visit(print)
        return True
    except Exception as e:
        print(f"文件 {file_path} 无效: {str(e)}")
        return False


def load_weights_without_decode(filepath, model):
    with h5py.File(filepath, mode='r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # 获取层名列表，兼容Python 3
        layer_names = f.attrs['layer_names']

        # 如果是字节类型，则进行解码
        if isinstance(layer_names[0], bytes):
            layer_names = [name.decode('utf8') for name in layer_names]

        layers = [model.get_layer(name) for name in layer_names]
        for layer in layers:
            g = f[layer.name]
            weights = [g[attr_name][()] for attr_name in g.attrs['weight_names']]
            layer.set_weights(weights)


def main():
    print(f"尝试加载模型: {MODEL_PATH}")
    assert os.path.isfile(MODEL_PATH), f"模型文件不存在: {MODEL_PATH}"

    # 检查H5文件
    if not check_h5_file(MODEL_PATH):
        print("请确认模型文件是否完整且未损坏")
        return

    # 尝试加载整个模型
    try:
        model = load_model(MODEL_PATH)
        print("成功加载整个模型")
    except Exception as e:
        print(f"加载整个模型失败: {str(e)}")
        print("尝试仅加载权重...")
        model = build_stixel_net()
        try:
            load_weights_without_decode(MODEL_PATH, model)
            print("成功加载模型权重")
        except Exception as e2:
            print(f"加载模型权重失败: {str(e2)}")
            return

    image_folder = "D:\Road Obstacle Detection\picture-for-test"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"找到 {len(image_files)} 张图片")

    for i, image_file in tqdm.tqdm(enumerate(image_files)):
        image_path = os.path.join(image_folder, image_file)
        img = cv2.imread(image_path)

        if img is not None:
            result = test_single_image(model, img)
            result_path = f"result_{os.path.splitext(image_file)[0]}.png"
            cv2.imwrite(result_path, result)
            print(f"已保存结果: {result_path}")


if __name__ == "__main__":
    main()