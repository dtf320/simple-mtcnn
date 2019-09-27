import cv2
import tensorflow as tf
from mtcnn import MTCNN, detection
import time


score_display = True
rect_color = (0,0,200)
text_color = (253,139,254)
landmark_color = (0,200,0)

mtcnn = MTCNN(trainable=False)
sess = tf.Session()
pfn, rfn, ofn = mtcnn.build(sess, restore=True)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
while ret:
    im = frame.copy()
    h,w = im.shape[:2]
    print(im.shape)
    ratio = 1
    im = cv2.resize(im,(int(ratio*w),int(ratio*h)))
    im = im[..., ::-1]

    start = time.time()
    face_axis, points, score = detection(pfn, rfn, ofn, im, detect_multiple_faces=True)

    fps = int(1/(time.time()-start))
    print('fps {}'.format(fps))
    im = im[..., ::-1]
    if len(face_axis) == 0:
        cv2.putText(im, 'no face, fps {}'.format(fps), (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    else:
        cv2.putText(im, '{} face, fps {}'.format(len(score),fps), (0, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        for idx, box in enumerate(face_axis):
            cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
            if score_display:
                cv2.putText(im, str(round(score[idx], 3)), (box[2], box[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        for p in points:
            for x, y in p:
                cv2.circle(im, (x, y), 1, color=landmark_color, thickness=-1)
    cv2.imshow('',im)
    cv2.waitKey(1)
    ret, frame = cap.read()

