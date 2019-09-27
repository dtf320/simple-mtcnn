import cv2
from mtcnn_detection import detection
import tensorflow as tf
from mtcnn import MTCNN
import os
import time
import shutil


if not os.path.exists('./result'):
    os.makedirs('./result')
mtcnn = MTCNN(trainable=False)
sess = tf.Session()
pfn, rfn, ofn = mtcnn.build(sess, restore=True)
data_path = './data/0--Parade/'

dump = {}

for img in os.listdir(data_path):
    img_path = os.path.join(data_path,img)
    im = cv2.imread(img_path)
    h,w = im.shape[:2]
    ratio = 1
    im = cv2.resize(im,(int(ratio*w),int(ratio*h)))
    im = im[..., ::-1]
    dump['img_name'] = img_path
    print(img_path)
    start = time.time()
    face_axis, points, score = detection(pfn, rfn, ofn, im, detect_multiple_faces=True)
    dump['face_axis'] = face_axis
    dump['points'] = points
    dump['score'] = score
    fps = 1/(time.time()-start)
    dump['fps'] = fps
    print('fps {}'.format(fps))
    img = im[..., ::-1]
    if len(face_axis) == 0:
        cv2.putText(img, 'no face, fps {}'.format(fps), (10,30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    else:
        cv2.putText(img, '{} face, fps {}'.format(len(score),fps), (10,30),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
        for idx, box in enumerate(face_axis):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0,0,255), thickness=1)
            cv2.putText(img, str(round(score[idx], 3)), (box[2], box[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)
        for p in points:
            for x, y in p:
                cv2.circle(img, (x, y), 1, color=(0,255,0), thickness=-1)

    cv2.imwrite(os.path.join('./result',os.path.split(img_path)[1]),img)
