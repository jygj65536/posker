import time
import sys
import logging
from tf_pose import common_new
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

w, h = (432, 638)
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))

# estimate human poses from a single image !
image = common_new.read_imgfile('./images/test_result_jy.pickle', None, None)
if image is None:
    logger.error('Image can not be read, path=%s' % './images/test_result_jy.pickle')
    sys.exit(-1)

t = time.time()
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
elapsed = time.time() - t

logger.info('inference image: %s in %.4f seconds.' % ('./images/test_result_jy.pickle', elapsed))

image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

try:
    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    a.set_title('Result')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    # show network output
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(bgimg, alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 2, 3)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 2, 4)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    plt.show()
except Exception as e:
    logger.warning('matplitlib error, %s' % e)
    cv2.imshow('result', image)
    cv2.waitKey()