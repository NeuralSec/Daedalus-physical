"""Demo for use yolo v3
"""
import os
import time
import cv2
import numpy as np
from model.yolo_model import YOLO
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import sys
sys.path.insert(0, '..') # Windows Path
import COCO.eval as Eval

#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

PATHS = ['adv_examples/plot/L2/f2/benign',
         'adv_examples/plot/L2/f2/0.3',
         'adv_examples/plot/L2/f2/0.7']


def transform_img(img):
    """Transform image by zooming in or out.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(416, 416, 3), processed image.
    """
    image = cv2.resize(img, (512,512))
    #roi = image[50:50+416, 50:50+416, :]
    #image = roi
    return image


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(416, 416, 3), processed image.
    """
    image = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)
        #print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        #print('box coordinate x,y,w,h: {0}'.format(box))
    print()

def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores, raw_out = yolo.predict(pimage, image.shape)
    #yolo._yolo.summary()
    #plot_model(yolo._yolo, to_file='model.png', show_shapes=True, show_layer_names=True)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)
        Eval.save_txt(f, image, boxes, classes, scores, all_classes)
    return image


def detect_vedio(video, yolo, all_classes):
    """Use yolo v3 to detect video.

    # Argument:
        video: video file.
        yolo: YOLO, yolo model.
        all_classes: all classes name.
    """
    camera = cv2.VideoCapture(video)
    cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

    while True:
        res, frame = camera.read()
        if not res:
            break
        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)
        if cv2.waitKey(110) & 0xff == 27:
                break
    camera.release()

def plot(paths):
    b_imgs = []
    b_r = []
    low_imgs = []
    l_r = []
    high_imgs =[]
    h_r = []
    for (root, dirs, files) in os.walk(paths[0]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                originalimgs = cv2.imread(path) # RGB image
                originalimgs = cv2.cvtColor(originalimgs, cv2.COLOR_BGR2RGB)
                originalimgs = cv2.resize(originalimgs, (416, 416), interpolation=cv2.INTER_CUBIC)
                o_img = process_image(originalimgs)
                b_imgs.extend(o_img)
                b_r.append(detect_image(originalimgs, yolo, all_classes))
    for (root, dirs, files) in os.walk(paths[1]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                lowconfimgs = cv2.imread(path) # RGB image
                lowconfimgs = cv2.cvtColor(lowconfimgs, cv2.COLOR_BGR2RGB)
                l_img = process_image(lowconfimgs)
                low_imgs.extend(l_img)
                l_r.append(detect_image(lowconfimgs, yolo, all_classes))
    for (root, dirs, files) in os.walk(paths[2]):
        if files:
            for f in files:
                path = os.path.join(root, f)
                highconfimgs = cv2.imread(path) # RGB image
                highconfimgs = cv2.cvtColor(highconfimgs, cv2.COLOR_BGR2RGB)
                h_img = process_image(highconfimgs)
                high_imgs.extend(h_img)
                h_r.append(detect_image(highconfimgs, yolo, all_classes))
    b_imgs = np.array(b_imgs)
    b_r = np.array(b_r)/255.
    low_imgs = np.array(low_imgs)
    l_r = np.array(l_r)/255.
    high_imgs = np.array(high_imgs)
    h_r = np.array(h_r)/255.
    print(b_imgs.shape, b_r.shape, low_imgs.shape, l_r.shape, high_imgs.shape, h_r.shape)
    results = np.stack((b_imgs, b_r, low_imgs, l_r, high_imgs, h_r))

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(6, 10, wspace=0.1, hspace=0.1)

    for i in range(6):
        for j in range(10):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(results[i, j], interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])
    gs.tight_layout(fig)
    plt.show()

if __name__ == '__main__':
    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)
    #plot(PATHS)

    #video = 'videos/test/Streetview.flv'
    #detect_vedio(video, yolo, all_classes)
    #'''
    # detect images in test floder.
    for (root, dirs, files) in os.walk('adv_examples/temp'):
        if files:
            for f in files:
                print(f)
                path = os.path.join(root, f)
                image = cv2.imread(path) # RGB image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = detect_image(image, yolo, all_classes)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite('detect_results/test/' + f, image)
    #'''
