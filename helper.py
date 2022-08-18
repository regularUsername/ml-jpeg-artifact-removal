import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np


def array2image(array, BGR=False):
    if BGR:
        return Image.fromarray(cv2.cvtColor((array * 255).astype("uint8"), cv2.COLOR_BGR2RGB))
    else:
        return Image.fromarray((array * 255).astype("uint8"))


def show_arrayimg(array, interpolation="bicubic"):
    plt.imshow(array,interpolation=interpolation)
    plt.show()


def imgfile2array(filename, size, normalize=True, scaler=Image.BOX):
    try:
        im = Image.open(filename)
        im = im.resize(size, scaler)
        im = im.convert(mode='RGB')
        im = np.asarray(im, "float32") / 255
        return im
    except Exception as e:
        print(f'failed loading: "{filename}" with error "{e}"')
        return None



def plot_model(hist, size=(16,4)):
    plt.figure(figsize=size)
    x = range(1, len(hist.epoch)+1)

    metrics = [x for x in hist.history.keys() if not "val_" in x]
    for i in range(len(metrics)):
        plt.subplot(1,len(metrics),i+1)
        plt.plot(x, hist.history[metrics[i]],'m--')
        if "val_"+metrics[i] in hist.history.keys():
            plt.plot(x, hist.history["val_"+metrics[i]], 'b-')
            plt.legend([metrics[i],"val_"+metrics[i]])
        else:
            plt.legend([metrics[i]])
        plt.xlabel('epoch')
        plt.ylim()
        plt.grid()

    plt.show()
