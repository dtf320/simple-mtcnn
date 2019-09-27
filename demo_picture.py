import time
from mtcnn import MTCNN,detection
import tensorflow as tf
import cv2
import os
from utils import resize_to_fixed_w

img_dir = './docs/imgs'
output_path = './docs/out'
score_display = True

rect_color = (0,0,200)
text_color = (253,139,254)
landmark_color = (0,200,0)

if not os.path.exists(output_path):
    os.makedirs(output_path)

mtcnn = MTCNN(trainable=False)
sess = tf.Session()
pfn,rfn,ofn = mtcnn.build(sess, restore=True)

for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir,img_file)
    im = cv2.imread(img_path)
    im = im[..., ::-1]
    h, w = im.shape[:2]

    start = time.time()
    face_axis, points, score = detection(pfn, rfn, ofn, im, detect_multiple_faces=True)
    fps = 1/(time.time() - start)

    im = im[..., ::-1]
    if len(face_axis) == 0:
        cv2.putText(im, 'no face', (0,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    else:
        for idx, box in enumerate(face_axis):
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
            if score_display:
                cv2.putText(im, str(round(score[idx], 3)), (box[2], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        for p in points:
            for x, y in p:
                cv2.circle(im, (x, y), 1, color=landmark_color, thickness=-1)
    # cv2.imshow('', im)
    # cv2.waitKey(0)
    img = resize_to_fixed_w(im,800)
    cv2.imwrite(os.path.join(output_path,img_file),img)