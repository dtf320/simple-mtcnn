import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from utils import bbreg,rerec,imresample,nms,generateBoundingBox,pad


class MTCNN:
    def __init__(self, trainable):
        self.trainable = trainable

    def build(self, sess, restore=False):
        input_p = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32, name='input')
        input_r = tf.placeholder(shape=(None, 24, 24, 3), dtype=tf.float32, name='input')
        input_o = tf.placeholder(shape=(None, 48, 48, 3), dtype=tf.float32, name='input')
        self.PNet(input_p)
        self.RNet(input_r)
        self.ONet(input_o)
        pnet = lambda img: sess.run(['pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'],
                                        feed_dict={input_p: img})
        rnet = lambda img: sess.run(['rnet/conv5-2/BiasAdd:0', 'rnet/prob1:0'],
                                        feed_dict={input_r: img})
        onet = lambda img: sess.run(['onet/conv6-2/BiasAdd:0', 'onet/conv6-3/BiasAdd:0', 'onet/prob1:0'],
                                        feed_dict={input_o: img})
        if restore:
            self.restore(sess)
        else:
            sess.run(tf.global_variables_initializer())
        return pnet,rnet,onet

    def prelu(self,inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = tf.get_variable('alpha', shape=(i,), trainable=self.trainable)
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keepdims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keepdims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax

    def restore(self, sess):
        import itertools
        assp = self.__vars_restore('pnet', 'det1.npy')
        assr = self.__vars_restore('rnet', 'det2.npy')
        asso = self.__vars_restore('onet', 'det3.npy')
        sess.run(list(itertools.chain(assp,assr,asso)))

    def __vars_restore(self, netname, npy):
        ass_list = []
        data = np.load(npy, encoding='latin1', allow_pickle=True).item()
        with tf.variable_scope(netname, reuse=True):
            for k in data.keys():
                for kk in data[k].keys():
                    tensor = tf.get_variable('{}/{}'.format(k.lower(), kk.lower()))
                    ass = tf.assign(tensor, data[k][kk])
                    ass_list.append(ass)
        return ass_list

    def PNet(self, inputs):
        with tf.variable_scope('pnet'):
            with slim.arg_scope([slim.conv2d],
                                padding='VALID',activation_fn=None):
                net = slim.conv2d(inputs,10,(3,3),scope='conv1')
                net = self.prelu(net,'prelu1')
                net = slim.max_pool2d(net,(2,2),2,padding='SAME')
                net = slim.conv2d(net,16,(3,3),scope='conv2')
                net = self.prelu(net,'prelu2')
                net = slim.conv2d(net,32,(3,3),scope='conv3')
                net = self.prelu(net,'prelu3')
                conv4_1 = slim.conv2d(net,2,(1,1),scope='conv4-1')
                prob1 = self.softmax(conv4_1,axis=3,name='prob1')
                conv4_2 = slim.conv2d(net,4,(1,1),scope='conv4-2')


    def RNet(self, inputs):
        with tf.variable_scope('rnet'):
            with slim.arg_scope([slim.conv2d],
                                padding='VALID',activation_fn=None):
                net = slim.conv2d(inputs,28,(3,3),scope='conv1')
                net = self.prelu(net,'prelu1')
                net = slim.max_pool2d(net,(3,3),2,padding='SAME')
                net = slim.conv2d(net,48,(3,3),scope='conv2')
                net = self.prelu(net, 'prelu2')
                net = slim.max_pool2d(net,(3,3),2,padding='VALID')
                net = slim.conv2d(net,64,(2,2),scope='conv3')
                net = self.prelu(net, 'prelu3')
                net = slim.flatten(net)
                conv4 = slim.fully_connected(net,128,activation_fn=None,scope='conv4')
                prelu4 = self.prelu(conv4,'prelu4')
                conv5_1 = slim.fully_connected(prelu4,2,activation_fn=None,scope='conv5-1')
                prob1 = self.softmax(conv5_1,axis=1,name='prob1')
                conv5_2 = slim.fully_connected(prelu4,4,activation_fn=None,scope='conv5-2')


    def ONet(self, inputs):
        with tf.variable_scope('onet'):
            with slim.arg_scope([slim.conv2d],
                                padding='VALID',activation_fn=None):
                net = slim.conv2d(inputs,32,(3,3),scope='conv1')
                net = self.prelu(net,'prelu1')
                net = slim.max_pool2d(net,(3,3),2,padding='SAME')
                net = slim.conv2d(net,64,(3,3),scope='conv2')
                net = slim.max_pool2d(net,(3,3),2, padding='VALID')
                net = self.prelu(net, 'prelu2')
                net = slim.conv2d(net,64,(3,3),scope='conv3')
                net = self.prelu(net, 'prelu3')
                net = slim.max_pool2d(net,(2,2),2,padding='SAME')
                net = slim.conv2d(net,128,(2,2),scope='conv4')
                net = self.prelu(net, 'prelu4')
                net = slim.flatten(net)
                conv4 = slim.fully_connected(net,256,activation_fn=None,scope='conv5')
                prelu5 = self.prelu(conv4,'prelu5')
                conv6_1 = slim.fully_connected(prelu5,2,activation_fn=None,scope='conv6-1')
                prob1 = self.softmax(conv6_1,axis=1,name='prob1')
                conv6_2 = slim.fully_connected(prelu5, 4, activation_fn=None, scope='conv6-2')
                conv6_3 = slim.fully_connected(prelu5, 10, activation_fn=None, scope='conv6-3')


def detection(pnet, rnet, onet, image, detect_multiple_faces=False, margin=0):
    """

    :param image: RGB
    :param detect_multiple_faces:
    :param margin: 截取人脸的时候边缘扩大的宽度，margin=20则输出的人脸上下左右各增加宽度10
    :return:
    """

    minsize = 12  # minimum size of face
    threshold = [0.9, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    factor_count = 0
    total_boxes = np.empty((0, 9))
    points = np.empty(0)
    h = image.shape[0]
    w = image.shape[1]
    minl = np.amin([h, w])
    m = 12.0 / minsize
    minl = minl * m
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales += [m * np.power(factor, factor_count)]
        minl = minl * factor
        factor_count += 1

    # first stage
    for scale in scales:
        hs = int(np.ceil(h * scale))
        ws = int(np.ceil(w * scale))
        im_data = imresample(image, (hs, ws))
        im_data = (im_data - 127.5) * 0.0078125
        img_x = np.expand_dims(im_data, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))
        out = pnet(img_y)
        out0 = np.transpose(out[0], (0, 2, 1, 3))
        out1 = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])

        # inter-scale nms
        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # second stage
        tempimg = np.zeros((24, 24, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = image[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (24, 24))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = rnet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        score = out1[1, :]
        ipass = np.where(score > threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = rerec(total_boxes.copy())

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # third stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
        tempimg = np.zeros((48, 48, 3, numbox))
        for k in range(0, numbox):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = image[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tempimg[:, :, :, k] = imresample(tmp, (48, 48))
            else:
                return np.empty()
        tempimg = (tempimg - 127.5) * 0.0078125
        tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
        out = onet(tempimg1)
        out0 = np.transpose(out[0])
        out1 = np.transpose(out[1])
        out2 = np.transpose(out[2])
        score = out2[1, :]
        points = out1
        ipass = np.where(score > threshold[2])
        points = points[:, ipass[0]]
        score_pass = score[ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
            pick = nms(total_boxes.copy(), 0.7, 'Min')
            score_pass = score_pass[pick]
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

    if points.size == 0:
        points = []
    else:
        points = np.split(points, points.shape[1], axis=1)
        points = [_.reshape(-1, 2, order='F') for _ in points]

    nrof_faces = total_boxes.shape[0]
    if nrof_faces == 0:
        return [], [], []

    # det --- [min_x,min_y,max_x,max_y]
    det = total_boxes[:, 0:4]
    face_pos = []
    img_size = np.asarray(image.shape)[0:2]
    if nrof_faces > 1:
        if detect_multiple_faces:
            for i in range(nrof_faces):
                face_pos.append(np.squeeze(det[i]))
        else:
            # w = det[:,2]-det[:,0]    h = det[:,3]-det[:,1]
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            # 保留最靠近图片中点的face
            index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
            face_pos.append(det[index, :])
    else:
        face_pos.append(np.squeeze(det))

    face_axis = []
    for i, det in enumerate(face_pos):
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        face_axis.append(bb)

    return face_axis, points, score_pass