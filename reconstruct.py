#!/usr/bin/env python3

import keras
from keras.models import load_model
from keras import backend as K
from itertools import product
from math import ceil
from PIL import Image
import sys
import numpy as np
from pathlib import Path

"""
requirements
keras
tensorflow or cntk as backend for keras
PIL
h5py
"""


def main():
    if len(sys.argv) <= 1:
        print("no filename entered (usage: reconstruct.py example.jpg")
        sys.exit()

    infile = Path(sys.argv[1])

    if len(sys.argv) > 2:
        outfile = Path(sys.argv[2])
    else:
        outfile = infile.with_name(infile.stem + "(reconstructed).png")

    im = process(infile)

    im.save(outfile)


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def process(imgpath) -> Image:
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        K.tensorflow_backend.set_session(tf.Session(config=config))

    model = load_model(
        'model2.hdf5', custom_objects={'PSNRLoss': PSNRLoss})

    p = 6  # overlap
    ts = 32
    ts2 = ts - p * 2  # actual tile_size

    im = np.asarray(Image.open(imgpath).convert('RGB')).astype('float32') / 255

    (h, w, _) = im.shape

    #neue abemssungen des bildes berechnen um auf vielfaches von ts zu kommen
    bh = ts2 * ceil((h + p * 2) / ts2)
    bw = ts2 * ceil((w + p * 2) / ts2)
    # links und oben nur overlap als rand und rechts und unten etwas mehr um auf vielfaches von ts zu kommen
    im = np.pad(im, ((p, p + bh - h), (p, p + bw - w), (0, 0)), mode='reflect')

    pred_im = np.zeros(im.shape, dtype="uint8")

    tiles = np.zeros(
        ((bw // ts2 + 1) * (bh // ts2 + 1), ts, ts, 3), dtype="float32")

    # bild in ts*ts große tiles aufteilen mit überlappung p
    for i, (x, y) in enumerate(product(range(0, bw, ts2), range(0, bh, ts2))):
        tiles[i] = im[y:(y + ts), x:(x + ts), ]

    #tiles durch cnn schicken
    pred_tiles = model.predict(tiles)

    # array für umwandlung in bild vorbereiten
    pred_tiles = np.clip(pred_tiles * 255, 0, 255).astype("uint8")

    # bild wieder zusammensetzen ohne überlappende bereiche
    for i, (x, y) in enumerate(product(range(p, bw, ts2), range(p, bh, ts2))):
        pred_im[y:(y + ts2), x:(x + ts2), ] = pred_tiles[i, p:-p, p:-p, ]

    # "padding" entfernen und array zu bild machen
    pred_im = Image.fromarray(pred_im[p:h + p, p:w + p, ])
    return pred_im


if __name__ == '__main__':
    main()
