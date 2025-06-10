#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse

import h5py
import numpy as np
import cv2
import tqdm as tqdm
from models import build_stixel_net
from data_loader import KittiStixelDataset
from albumentations import (
    Compose,
    Resize,
    Normalize,
)
import tensorflow.keras.backend as K

# 固定模型路径
MODEL_PATH = "D:\Road Obstacle Detection\saved_models\model.h5"

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


def load_weights_without_decode(filepath, model):
    with h5py.File(filepath, mode='r') as f:
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # Convert byte strings to regular strings
        layer_names_bytes = f.attrs['layer_names']
        layer_names = [name.decode('utf8') for name in layer_names_bytes]

        layers = [model.get_layer(name) for name in layer_names]
        for layer in layers:
            g = f[layer.name]
            weights = [g[attr_name][()] for attr_name in g.attrs['weight_names']]
            layer.set_weights(weights)

def main():
    assert os.path.isfile(MODEL_PATH)
    from config import Config

    dt_config = Config()
    model = build_stixel_net()
    load_weights_without_decode(MODEL_PATH, model)
    #model.load_weights(MODEL_PATH)
    selected_series = [
        "2011_09_26_drive_0002_sync",
        "2011_09_26_drive_0009_sync",
        "2011_09_26_drive_0013_sync",
        "2011_09_26_drive_0064_sync"
    ]

    val_set = KittiStixelDataset(
        data_path=dt_config.DATA_PATH,
        ground_truth_path=dt_config.GROUND_TRUTH_PATH,
        phase="val",
        batch_size=1,
        input_shape=None,
        selected_series=selected_series  # 传入指定的序列
    )

    indices = range(len(val_set._image_paths))  # 遍历所有筛选后的图片
    for i, idx in tqdm.tqdm(enumerate(indices)):
        img, _ = val_set[idx]
        img = img[0]

        result = test_single_image(model, img)
        cv2.imwrite("result{}.png".format(i), result)


if __name__ == "__main__":
    main()