import os
os.environ['force_plaidML'] = '1'

import sys
import argparse
from utils import Path_utils
from utils import os_utils
from facelib import LandmarksProcessor
from pathlib import Path
import numpy as np
import cv2
import time
import multiprocessing
import threading
import traceback
from tqdm import tqdm
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from utils.cv2_utils import *
from utils import image_utils
import shutil



def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T

def random_transform(image, rotation_range=10, zoom_range=0.5, shift_range=0.05, random_flip=0):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(
        image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result

# get pair of random warped images from aligned face image
def random_warp(image, coverage=160, scale = 5, zoom = 1):
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - coverage//2, 128 + coverage//2, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + np.random.normal(size=(5,5), scale=scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=scale)

    interp_mapx = cv2.resize(mapx, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')
    interp_mapy = cv2.resize(mapy, (80*zoom,80*zoom))[8*zoom:72*zoom,8*zoom:72*zoom].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    src_points = np.stack([mapx.ravel(), mapy.ravel() ], axis=-1)
    dst_points = np.mgrid[0:65*zoom:16*zoom,0:65*zoom:16*zoom].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    target_image = cv2.warpAffine(image, mat, (64*zoom,64*zoom))

    return warped_image, target_image

def input_process(stdin_fd, sq, str):
    sys.stdin = os.fdopen(stdin_fd)
    try:
        inp = input (str)
        sq.put (True)
    except:
        sq.put (False)

def input_in_time (str, max_time_sec):
    sq = multiprocessing.Queue()
    p = multiprocessing.Process(target=input_process, args=( sys.stdin.fileno(), sq, str))
    p.start()
    t = time.time()
    inp = False
    while True:
        if not sq.empty():
            inp = sq.get()
            break
        if time.time() - t > max_time_sec:
            break
    p.terminate()
    sys.stdin = os.fdopen( sys.stdin.fileno() )
    return inp



def subprocess(sq,cq):
    prefetch = 2
    while True:
        while prefetch > -1:
            cq.put ( np.array([1]) ) #memory leak numpy==1.16.0 , but all fine in 1.15.4
            #cq.put ( [1] )  #no memory leak
            prefetch -= 1

        sq.get() #waiting msg from serv to continue posting
        prefetch += 1



def get_image_hull_mask (image_shape, image_landmarks):
    if len(image_landmarks) != 68:
        raise Exception('get_image_hull_mask works only with 68 landmarks')

    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    cv2.fillConvexPoly( hull_mask, cv2.convexHull( np.concatenate ( (image_landmarks[0:17], image_landmarks[48:], [image_landmarks[0]], [image_landmarks[8]], [image_landmarks[16]]))    ), (1,) )
    cv2.fillConvexPoly( hull_mask, cv2.convexHull( np.concatenate ( (image_landmarks[27:31], [image_landmarks[33]]) )                                                                    ), (1,) )
    cv2.fillConvexPoly( hull_mask, cv2.convexHull( np.concatenate ( (image_landmarks[17:27], [image_landmarks[0]], [image_landmarks[27]], [image_landmarks[16]], [image_landmarks[33]])) ), (1,) )

    return hull_mask


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T

#mean_face_x = np.array([
#0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
#0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
#0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
#0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
#0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
#0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
#0.553364, 0.490127, 0.42689 ])
#
#mean_face_y = np.array([
#0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
#0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
#0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
#0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
#0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
#0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
#0.784792, 0.824182, 0.831803, 0.824182 ])
#
#landmarks_2D = np.stack( [ mean_face_x, mean_face_y ], axis=1 )

def get_transform_mat (image_landmarks, output_size, scale=1.0):
    if not isinstance(image_landmarks, np.ndarray):
        image_landmarks = np.array (image_landmarks)

    padding = (output_size / 64) * 12

    mat = umeyama(image_landmarks[17:], landmarks_2D, True)[0:2]
    mat = mat * (output_size - 2 * padding)
    mat[:,2] += padding
    mat *= (1 / scale)
    mat[:,2] += -output_size*( ( (1 / scale) - 1.0 ) / 2 )

    return mat

#alignments = []
#
#aligned_path_image_paths = Path_utils.get_image_paths("D:\\DeepFaceLab\\workspace issue\\data_dst\\aligned")
#for filepath in tqdm(aligned_path_image_paths, desc="Collecting alignments", ascii=True ):
#    filepath = Path(filepath)
#
#    if filepath.suffix == '.png':
#        dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
#    elif filepath.suffix == '.jpg':
#        dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
#    else:
#        print ("%s is not a dfl image file" % (filepath.name) )
#
#    #source_filename_stem = Path( dflimg.get_source_filename() ).stem
#    #if source_filename_stem not in alignments.keys():
#    #    alignments[ source_filename_stem ] = []
#
#    #alignments[ source_filename_stem ].append (dflimg.get_source_landmarks())
#    alignments.append (dflimg.get_source_landmarks())
import string

def main():
    from nnlib import nnlib
    exec( nnlib.import_all( device_config=nnlib.device.Config() ), locals(), globals() )
    PMLTile = nnlib.PMLTile
    PMLK = nnlib.PMLK

    import tensorflow as tf
    tfkeras = tf.keras
    tfK = tfkeras.backend
    lin = np.broadcast_to( np.linspace(1,10,10), (10,10) )

    #a = np.broadcast_to ( np.concatenate( [np.linspace(1,4,4), np.linspace(5,1,5)] ), (9,9) )
    #a = (a + a.T)-1
    #a = a[np.newaxis,:,:,np.newaxis]

    class ReflectionPadding2D():
        class TileOP(PMLTile.Operation):
            def __init__(self, input, h_pad, w_pad):
                if K.image_data_format() == 'channels_last':
                    if input.shape.ndims == 4:
                        H, W = input.shape.dims[1:3]
                        if (type(H) == int and h_pad >= H) or \
                        (type(W) == int and w_pad >= W):
                            raise ValueError("Paddings must be less than dimensions.")

                        c = """ function (I[B, H, W, C] ) -> (O) {{
                                WE = W + {w_pad}*2;
                                HE = H + {h_pad}*2;
                            """.format(h_pad=h_pad, w_pad=w_pad)
                        if w_pad > 0:
                            c += """
                                LEFT_PAD [b, h, w , c : B, H, WE, C ] = =(I[b, h, {w_pad}-w,            c]), w < {w_pad} ;
                                HCENTER  [b, h, w , c : B, H, WE, C ] = =(I[b, h, w-{w_pad},            c]), w < W+{w_pad}-1 ;
                                RIGHT_PAD[b, h, w , c : B, H, WE, C ] = =(I[b, h, 2*W - (w-{w_pad}) -2, c]);
                                LCR = LEFT_PAD+HCENTER+RIGHT_PAD;
                            """.format(h_pad=h_pad, w_pad=w_pad)
                        else:
                            c += "LCR = I;"

                        if h_pad > 0:
                            c += """
                                TOP_PAD   [b, h, w , c : B, HE, WE, C ] = =(LCR[b, {h_pad}-h,            w, c]), h < {h_pad};
                                VCENTER   [b, h, w , c : B, HE, WE, C ] = =(LCR[b, h-{h_pad},            w, c]), h < H+{h_pad}-1 ;
                                BOTTOM_PAD[b, h, w , c : B, HE, WE, C ] = =(LCR[b, 2*H - (h-{h_pad}) -2, w, c]);
                                TVB = TOP_PAD+VCENTER+BOTTOM_PAD;
                            """.format(h_pad=h_pad, w_pad=w_pad)
                        else:
                            c += "TVB = LCR;"

                        c += "O = TVB; }"

                        inp_dims = input.shape.dims
                        out_dims = (inp_dims[0], inp_dims[1]+h_pad*2, inp_dims[2]+w_pad*2, inp_dims[3])
                    else:
                        raise NotImplemented
                else:
                    raise NotImplemented

                super(ReflectionPadding2D.TileOP, self).__init__(c, [('I', input) ],
                        [('O', PMLTile.Shape(input.shape.dtype, out_dims ) )])

        def __init__(self, h_pad, w_pad):
            self.h_pad, self.w_pad = h_pad, w_pad

        def __call__(self, inp):
            return ReflectionPadding2D.TileOP.function(inp, self.h_pad, self.w_pad)

    sh_w = 9
    sh_h = 9
    sh = (1,sh_h,sh_w,1)
    w_pad, h_pad = 8,8


    t1 = tfK.placeholder (sh )
    t2 = tf.pad(t1, [ [0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

    pt1 = K.placeholder (sh )
    pt2 = ReflectionPadding2D(h_pad, w_pad )(pt1)

    tfunc = tfK.function ([t1],[t2])
    ptfunc = K.function([pt1],[pt2])

    for i in range(100):
        a = np.random.uniform( size=sh)
        # a = np.broadcast_to ( np.concatenate( [np.linspace(1,4,4), np.linspace(5,1,5)] ), (9,9) )
        # a = np.broadcast_to (np.linspace(1,9,9), (9,9) )
        # a = (a + a.T)-1
        # a = a[np.newaxis,:,:,np.newaxis]

        t = tfunc([a]) [0][0,:,:,0]
        pt = ptfunc ([a])[0][0,:,:,0]
        if np.allclose(t, pt):
            print ("all_close = True")
        else:
            print ("all_close = False\r\n")
            print(t,"")
            print(pt)
    import code
    code.interact(local=dict(globals(), **locals()))

    image = cv2.imread(r'D:\DFLBuild\test\00000.png').astype(np.float32) / 255.0
    image = np.expand_dims (image, 0)
    image_shape = image.shape

    image2 = cv2.imread(r'D:\DFLBuild\test\00001.png').astype(np.float32) / 255.0
    image2 = np.expand_dims (image2, 0)
    image2_shape = image2.shape


    # class ReflectionPadding2D():
    #     def __init__(self, h_pad, w_pad):
    #         self.h_pad, self.w_pad = h_pad, w_pad

    #     def __call__(self, inp):
    #         h_pad, w_pad = self.h_pad, self.w_pad
    #         if K.image_data_format() == 'channels_last':
    #             if inp.shape.ndims == 4:
    #                 w = K.concatenate ([ inp[:,:,w_pad:0:-1,:],
    #                                      inp,
    #                                      inp[:,:,-2:-w_pad-2:-1,:] ], axis=2 )

    #                 h = K.concatenate ([ w[:,h_pad:0:-1,:,:],
    #                                      w,
    #                                      w[:,-2:-h_pad-2:-1,:,:] ], axis=1 )

    #                 return h
    #             else:
    #                 raise NotImplemented
    #         else:
    #             raise NotImplemented
    #f = ReflectionPadding2D.function(t, [1,65,65,1], [1,1,1,1], [1,1,1,1])

    #x, = K.function ([t],[f]) ([ image ])

    #image = np.random.uniform ( size=(1,256,256,3) )
    #image2 = np.random.uniform ( size=(1,256,256,3) )

    #t1 = K.placeholder ( (None,) + image_shape[1:], name="t1" )
    #t2 = K.placeholder ( (None,None,None,None), name="t2" )

    #l1_t = DSSIMObjective() (t1,t2 )
    #l1, = K.function([t1, t2],[l1_t]) ([image, image2])
    #
    #print (l1)
    #t[:,0:64,64::2,:].source.op.code
    """
t1[:,0:64,64::2,128:]

function (I[N0, N1, N2, N3]) -> (O)
 {\n
     Start0 = max(0, 0);
     Offset0 = Start0;
 O[
     i0, i1, i2, i3:
     ((N0 - (Start0)) + 1 - 1)/1,
     (64 + 1 - 1)/1,
     (192 + 2 - 1)/2,
     (-125 + 1 - 1)/1]
     =
     =(I[1*i0+Offset0, 1*i1+0, 2*i2+64, 1*i3+128]);

                    }

    """
    import code
    code.interact(local=dict(globals(), **locals()))


    #==============================================================
    from nnlib import nnlib
    exec( nnlib.import_all( device_config=nnlib.device.Config(cpu_only=True) ), locals(), globals() )
    #tf = nnlib.tf
    #
    #model_path= Path(__file__).parent / 's3fd.pb'
    #
    #with tf.gfile.GFile( str(model_path), "rb") as gfile:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(gfile.read())
    #fa_graph = tf.Graph()
    #with fa_graph.as_default():
    #    tf.import_graph_def(graph_def, name="s3fd")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def t2kw_conv2d (src):
        if src.bias is not None:
            return [ np.moveaxis(src.weight.data.cpu().numpy(), [0,1,2,3], [3,2,0,1]), src.bias.data.cpu().numpy() ]
        else:
            return [ np.moveaxis(src.weight.data.cpu().numpy(), [0,1,2,3], [3,2,0,1])]


    def t2kw_bn2d(src):
        return [ src.weight.data.cpu().numpy(), src.bias.data.cpu().numpy(), src.running_mean.cpu().numpy(), src.running_var.cpu().numpy() ]


    def sf3d_keras(input_shape, sf3d_torch):

        inp = Input ( (None, None,3), dtype=K.floatx() )
        x = inp
        x = Lambda ( lambda x: x - K.constant([104,117,123]), output_shape=(None,None,3) ) (x)

        x = Conv2D(64, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv1_1), activation='relu') (x)
        x = Conv2D(64, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv1_2), activation='relu') (x)
        x = MaxPooling2D()(x)

        x = Conv2D(128, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv2_1), activation='relu') (x)
        x = Conv2D(128, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv2_2), activation='relu') (x)
        x = MaxPooling2D()(x)

        x = Conv2D(256, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv3_1), activation='relu') (x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv3_2), activation='relu') (x)
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv3_3), activation='relu') (x)
        f3_3 = x
        x = MaxPooling2D()(x)

        x = Conv2D(512, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv4_1), activation='relu') (x)
        x = Conv2D(512, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv4_2), activation='relu') (x)
        x = Conv2D(512, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv4_3), activation='relu') (x)
        f4_3 = x
        x = MaxPooling2D()(x)

        x = Conv2D(512, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv5_1), activation='relu') (x)
        x = Conv2D(512, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv5_2), activation='relu') (x)
        x = Conv2D(512, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv5_3), activation='relu') (x)
        f5_3 = x
        x = MaxPooling2D()(x)

        x = ZeroPadding2D(padding=(3,3))(x)
        x = Conv2D(1024, kernel_size=3, strides=1, padding='valid', weights=t2kw_conv2d(sf3d_torch.fc6), activation='relu') (x)
        x = Conv2D(1024, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d(sf3d_torch.fc7), activation='relu') (x)
        ffc7 = x

        x = Conv2D(256, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d(sf3d_torch.conv6_1), activation='relu') (x)
        x = ZeroPadding2D(padding=(1,1))(x)
        x = Conv2D(512, kernel_size=3, strides=2, padding='valid', weights=t2kw_conv2d(sf3d_torch.conv6_2), activation='relu') (x)
        f6_2 = x

        x = Conv2D(128, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d(sf3d_torch.conv7_1), activation='relu') (x)
        x = ZeroPadding2D(padding=(1,1))(x)
        x = Conv2D(256, kernel_size=3, strides=2, padding='valid', weights=t2kw_conv2d(sf3d_torch.conv7_2), activation='relu') (x)
        f7_2 = x

        def L2Norm(n_channels, scale=1.0, torch_weights=None):
            def func(x):
                weights = torch_weights.weight.data.cpu().numpy()
                x = Lambda ( lambda x: \
                                      x / (K.sqrt( K.sum( K.pow(x, 2), axis=-1, keepdims=True ) ) + 1e-10) \
                                        * K.reshape ( K.constant(weights), (1,1,n_channels))
                            , output_shape=(None,None,n_channels)) (x)
                return x
            return func

        f3_3 = L2Norm(256, scale=10, torch_weights=sf3d_torch.conv3_3_norm)(f3_3)
        f4_3 = L2Norm(512, scale=8, torch_weights=sf3d_torch.conv4_3_norm)(f4_3)
        f5_3 = L2Norm(512, scale=5, torch_weights=sf3d_torch.conv5_3_norm)(f5_3)

        cls1 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv3_3_norm_mbox_conf), activation='softmax')(f3_3)
        reg1 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv3_3_norm_mbox_loc)) (f3_3)

        cls2 = Conv2D(2, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv4_3_norm_mbox_conf), activation='softmax')(f4_3)
        reg2 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv4_3_norm_mbox_loc)) (f4_3)

        cls3 = Conv2D(2, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv5_3_norm_mbox_conf), activation='softmax')(f5_3)
        reg3 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv5_3_norm_mbox_loc)) (f5_3)

        cls4 = Conv2D(2, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.fc7_mbox_conf), activation='softmax')(ffc7)
        reg4 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.fc7_mbox_loc)) (ffc7)

        cls5 = Conv2D(2, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv6_2_mbox_conf), activation='softmax')(f6_2)
        reg5 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv6_2_mbox_loc)) (f6_2)

        cls6 = Conv2D(2, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv7_2_mbox_conf), activation='softmax')(f7_2)
        reg6 = Conv2D(4, kernel_size=3, strides=1, padding='same', weights=t2kw_conv2d(sf3d_torch.conv7_2_mbox_loc)) (f7_2)

        L = Lambda ( lambda x: x[:,:,:,-1], output_shape=(None,None,1) )
        cls1, cls2, cls3, cls4, cls5, cls6 = [ L(x) for x in [cls1, cls2, cls3, cls4, cls5, cls6] ]

        return Model(inp, [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6])

    class L2Norm(nn.Module):
        def __init__(self, n_channels, scale=1.0):
            super(L2Norm, self).__init__()
            self.n_channels = n_channels
            self.scale = scale
            self.eps = 1e-10
            self.weight = nn.Parameter(torch.Tensor(self.n_channels))
            self.weight.data *= 0.0
            self.weight.data += self.scale

        def forward(self, x):
            norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
            x = x / norm * self.weight.view(1, -1, 1, 1)
            return x


    class s3fd_torch(nn.Module):
        def __init__(self):
            super(s3fd_torch, self).__init__()
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

            self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
            self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

            self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
            self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

            self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
            self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

            self.conv3_3_norm = L2Norm(256, scale=10)
            self.conv4_3_norm = L2Norm(512, scale=8)
            self.conv5_3_norm = L2Norm(512, scale=5)

            self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
            self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
            self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
            self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
            self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
            self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

            self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
            self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
            self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
            self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
            self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
            self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

        def forward(self, x):

            h = F.relu(self.conv1_1(x))
            h = F.relu(self.conv1_2(h))
            h = F.max_pool2d(h, 2, 2)

            h = F.relu(self.conv2_1(h))
            h = F.relu(self.conv2_2(h))
            h = F.max_pool2d(h, 2, 2)

            h = F.relu(self.conv3_1(h))
            h = F.relu(self.conv3_2(h))
            h = F.relu(self.conv3_3(h))
            f3_3 = h
            h = F.max_pool2d(h, 2, 2)

            h = F.relu(self.conv4_1(h))
            h = F.relu(self.conv4_2(h))
            h = F.relu(self.conv4_3(h))
            f4_3 = h
            h = F.max_pool2d(h, 2, 2)

            h = F.relu(self.conv5_1(h))
            h = F.relu(self.conv5_2(h))
            h = F.relu(self.conv5_3(h))
            f5_3 = h
            h = F.max_pool2d(h, 2, 2)

            h = F.relu(self.fc6(h))
            h = F.relu(self.fc7(h))
            ffc7 = h

            h = F.relu(self.conv6_1(h))

            h = F.relu(self.conv6_2(h))

            f6_2 = h

            h = F.relu(self.conv7_1(h))
            h = F.relu(self.conv7_2(h))
            f7_2 = h


            f3_3 = self.conv3_3_norm(f3_3)
            f4_3 = self.conv4_3_norm(f4_3)
            f5_3 = self.conv5_3_norm(f5_3)

            cls1 = self.conv3_3_norm_mbox_conf(f3_3)
            reg1 = self.conv3_3_norm_mbox_loc(f3_3)
            cls2 = self.conv4_3_norm_mbox_conf(f4_3)
            reg2 = self.conv4_3_norm_mbox_loc(f4_3)
            cls3 = self.conv5_3_norm_mbox_conf(f5_3)
            reg3 = self.conv5_3_norm_mbox_loc(f5_3)

            cls4 = self.fc7_mbox_conf(ffc7)
            reg4 = self.fc7_mbox_loc(ffc7)
            cls5 = self.conv6_2_mbox_conf(f6_2)
            reg5 = self.conv6_2_mbox_loc(f6_2)
            cls6 = self.conv7_2_mbox_conf(f7_2)
            reg6 = self.conv7_2_mbox_loc(f7_2)

            # max-out background label
            chunk = torch.chunk(cls1, 4, 1)
            bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
            cls1 = torch.cat ([bmax,chunk[3]], dim=1)
            cls1, cls2, cls3, cls4, cls5, cls6 = [ F.softmax(x, dim=1) for x in [cls1, cls2, cls3, cls4, cls5, cls6] ]
            return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

    model_path = r"D:\DFLBuild\test\s3fd.pth"
    model_weights = torch.load(str(model_path))

    device = 'cpu'
    fd_torch = s3fd_torch()
    fd_torch.load_state_dict(model_weights)
    fd_torch.eval()

    def decode(loc, priors, variances):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
                               1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def softmax(x, axis=-1):
        y = np.exp(x - np.max(x, axis, keepdims=True))
        return y / np.sum(y, axis, keepdims=True)

    def nms(dets, thresh):
        """ Perform Non-Maximum Suppression """
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect_torch(olist):

        bboxlist = []

        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            poss = [*zip(*np.where(ocls[:, 1, :, :] > 0.05))]

            #import code
            #code.interact(local=dict(globals(), **locals()))

            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = np.ascontiguousarray(oreg[0, :, hindex, windex]).reshape((1, 4))
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))

        return bboxlist

    def detect_keras(olist):
        bboxlist = []
        for i, ((ocls,), (oreg,)) in enumerate ( zip ( olist[::2], olist[1::2] ) ):
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            s_d2 = stride / 2
            s_m4 = stride * 4

            for hindex, windex in zip(*np.where(ocls > 0.05)):
                score = ocls[hindex, windex]
                loc   = oreg[hindex, windex, :]
                priors = np.array([windex * stride + s_d2, hindex * stride + s_d2, s_m4, s_m4])
                priors_2p = priors[2:]
                box = np.concatenate((priors[:2] + loc[:2] * 0.1 * priors_2p,
                                      priors_2p * np.exp(loc[2:] * 0.2)) )
                box[:2] -= box[2:] / 2
                box[2:] += box[:2]

                bboxlist.append([*box, score])

        bboxlist = np.array(bboxlist)
        if len(bboxlist) == 0:
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[nms(bboxlist, 0.3), :]
        bboxlist = [ x[:-1] for x in bboxlist if x[-1] >= 0.5]
        return bboxlist

    #img = np.random.uniform ( size=(480,270,3) ) * 255
    img = cv2.imread ( r"D:\DFLBuild\test\00000.png" )

    torch_img = torch.from_numpy( np.expand_dims((img - np.array([104, 117, 123])).transpose(2, 0, 1),0) ).float()

    import time
    t = time.time()
    olist_torch = [x.data.cpu().numpy() for x in fd_torch( torch.autograd.Variable( torch_img, volatile=True) )]
    print ("torch took:", time.time() - t)

    fd_keras_path = r"D:\DFLBuild\test\S3FD.h5"
    if Path(fd_keras_path).exists():
        fd_keras = keras.models.load_model (fd_keras_path)
    else:
        fd_keras = sf3d_keras(img.shape, fd_torch)
        fd_keras.compile ( loss='mse', optimizer='adam' )
        fd_keras.save ( fd_keras_path )

    t = time.time()
    olist_keras = fd_keras.predict( np.expand_dims(img,0) )
    print ("keras took:", time.time() - t)
    #abs_diff = 0
    #for i in range(len(olist_torch)):
    #    td = np.transpose( olist_torch[i], (0,2,3,1) )
    #    kd = olist_keras[i]
    #    td = td[...,-1]
    #    kd = kd[...,-1]
    #    p = np.ndarray.flatten(td-kd)
    #    diff = np.sum ( np.abs(p))
    #    print ("nparams=", len(p), " diff=",diff, "diff_per_param=", diff / len(p)  )
    #    abs_diff += diff
    #import code
    #code.interact(local=dict(globals(), **locals()))
    #print ("Total absolute diff = ", abs_diff)

    t = time.time()
    bbox_torch = detect_torch(olist_torch)
    bbox_torch = bbox_torch[ nms(bbox_torch, 0.3) , :]
    bbox_torch = [x for x in bbox_torch if x[-1] >= 0.5]
    print ("torch took:", time.time() - t)

    t = time.time()
    bbox_keras = detect_keras(olist_keras)
    print ("keras took:", time.time() - t)

    #bbox_keras = bbox_keras[ nms(bbox_keras, 0.3) , :]
    #bbox_keras = [x for x in bbox_keras if x[-1] >= 0.5]

    print (bbox_torch)
    print (bbox_keras)

    import code
    code.interact(local=dict(globals(), **locals()))
    #===============================================================================



    '''
    >>> t[:,0:64,64::2,:].source.op.code
function (I[N0, N1, N2, N3]) -> (O) {

O[i0, i1, i2, i3: (1 + 1 - 1)/1, (64 + 1 - 1)/1, (64 + 2 - 1)/2, (1 + 1 - 1)/1] =
       =(I[1*i0+0, 1*i1+0, 2*i2+64, 1*i3+0]);


        Status GetWindowedOutputSizeVerboseV2(int64 input_size, int64 filter_size,
                                          int64 dilation_rate, int64 stride,
                                          Padding padding_type, int64* output_size,
                                          int64* padding_before,
                                          int64* padding_after) {
      if (stride <= 0) {
        return errors::InvalidArgument("Stride must be > 0, but got ", stride);
      }
      if (dilation_rate < 1) {
        return errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                       dilation_rate);
      }

      // See also the parallel implementation in GetWindowedOutputSizeFromDimsV2.
      int64 effective_filter_size = (filter_size - 1) * dilation_rate + 1;
      switch (padding_type) {
        case Padding::VALID:
          *output_size = (input_size - effective_filter_size + stride) / stride;
          *padding_before = *padding_after = 0;
          break;
        case Padding::EXPLICIT:
          *output_size = (input_size + *padding_before + *padding_after -
                          effective_filter_size + stride) /
                         stride;
          break;
        case Padding::SAME:
          *output_size = (input_size + stride - 1) / stride;
          const int64 padding_needed =
              std::max(int64{0}, (*output_size - 1) * stride +
                                     effective_filter_size - input_size);
          // For odd values of total padding, add more padding at the 'right'
          // side of the given dimension.
          *padding_before = padding_needed / 2;
          *padding_after = padding_needed - *padding_before;
          break;
      }
      if (*output_size < 0) {
        return errors::InvalidArgument(
            "Computed output size would be negative: ", *output_size,
            " [input_size: ", input_size,
            ", effective_filter_size: ", effective_filter_size,
            ", stride: ", stride, "]");
      }
      return Status::OK();
    }
    '''
    class ExtractImagePatchesOP(PMLTile.Operation):
        def __init__(self, input, ksizes, strides, rates, padding='valid'):

            batch, in_rows, in_cols, depth = input.shape.dims

            ksize_rows = ksizes[1];
            ksize_cols = ksizes[2];

            stride_rows = strides[1];
            stride_cols = strides[2];

            rate_rows = rates[1];
            rate_cols = rates[2];

            ksize_rows_eff = ksize_rows + (ksize_rows - 1) * (rate_rows - 1);
            ksize_cols_eff = ksize_cols + (ksize_cols - 1) * (rate_cols - 1);

            #if padding == 'valid':

            out_rows = (in_rows - ksize_rows_eff + stride_rows) / stride_rows;
            out_cols = (in_cols - ksize_cols_eff + stride_cols) / stride_cols;

            out_sizes = (batch, out_rows, out_cols, ksize_rows * ksize_cols * depth);



            B, H, W, CI = input.shape.dims

            RATE = PMLK.constant ([1,rate,rate,1], dtype=PMLK.floatx() )

            #print (target_dims)
            code = """function (I[B, {H}, {W}, {CI} ], RATES[RB, RH, RW, RC] ) -> (O) {

                        O[b, {wnd_size}, {wnd_size}, ] = =(I[b, h, w, ci]);

                    }""".format(H=H, W=W, CI=CI, RATES=rates, wnd_size=wnd_size)

            super(ExtractImagePatchesOP, self).__init__(code, [('I', input) ],
                    [('O', PMLTile.Shape(input.shape.dtype, out_sizes ) )])




    f = ExtractImagePatchesOP.function(t, [1,65,65,1], [1,1,1,1], [1,1,1,1])

    x, = K.function ([t],[f]) ([ image ])
    print(x.shape)

    import code
    code.interact(local=dict(globals(), **locals()))

    #from nnlib import nnlib
    #exec( nnlib.import_all( device_config=nnlib.device.Config(cpu_only=True) ), locals(), globals() )
    #
    #rnd_data = np.random.uniform( size=(1,64,64,3) )
    #bgr_shape = (64, 64, 3)
    #input_layer = Input(bgr_shape)
    #x = input_layer
    #x = Conv2D(64, 3, padding='same')(x)
    #x = Conv2D(128, 3, padding='same')(x)
    #x = Conv2D(256, 3, padding='same')(x)
    #x = Conv2D(512, 3, padding='same')(x)
    #x = Conv2D(1024, 3, padding='same')(x)
    #x = Conv2D(3, 3, padding='same')(x)
    #
    #model = Model (input_layer, [x])
    #model.compile(optimizer=Adam(), loss='mse')
    #model.train_on_batch ([rnd_data], [rnd_data])
    ##model.save (r"D:\DFLBuild\test\test_model.h5")
    #
    #import code
    #code.interact(local=dict(globals(), **locals()))



    import ffmpeg

    path = Path('D:/deepfacelab/test')
    input_path = str(path / 'input.mp4')
    #stream = ffmpeg.input(str(path / 'input.mp4') )
    #stream = ffmpeg.hflip(stream)
    #stream = ffmpeg.output(stream, str(path / 'output.mp4') )
    #ffmpeg.run(stream)
    (
        ffmpeg
        .input( str(path / 'input.mp4'))
        .hflip()
        .output( str(path / 'output.mp4'), r="23000/1001" )
        .run()
    )

    #probe = ffmpeg.probe(str(path / 'input.mp4'))

    #out, _ = (
    #    ffmpeg
    #    .input( input_path )
    #    .output('pipe:', format='rawvideo', pix_fmt='rgb24')
    #    .run(capture_stdout=True)
    #)
    #video = (
    #    np
    #    .frombuffer(out, np.uint8)
    #    .reshape([-1, height, width, 3])
    #)

    import code
    code.interact(local=dict(globals(), **locals()))




    from nnlib import nnlib
    exec( nnlib.import_all(), locals(), globals() )

    #ch = 3
    #def softmax(x, axis=-1): #from K numpy backend
    #    y = np.exp(x - np.max(x, axis, keepdims=True))
    #    return y / np.sum(y, axis, keepdims=True)
    #
    #def gauss_kernel(size, sigma):
    #    coords = np.arange(0,size, dtype=K.floatx() )
    #    coords -= (size - 1 ) / 2.0
    #    g = coords**2
    #    g *= ( -0.5 / (sigma**2) )
    #    g = np.reshape (g, (1,-1)) + np.reshape(g, (-1,1) )
    #    g = np.reshape (g, (1,-1))
    #    g = softmax(g)
    #    g = np.reshape (g, (size, size, 1, 1))
    #    g = np.tile (g, (1,1,ch, size*size*ch))
    #    return K.constant(g, dtype=K.floatx() )
    #
    ##kernel = gauss_kernel(11,1.5)
    #kernel = K.constant( np.ones ( (246,246, 3, 1) ) , dtype=K.floatx() )
    ##g = np.eye(9).reshape((3, 3, 1, 9))
    ##g = np.tile (g, (1,1,3,1))
    ##kernel = K.constant(g , dtype=K.floatx() )
    #
    #def reducer(x):
    #    shape = K.shape(x)
    #    x = K.reshape(x, (-1, shape[-3] , shape[-2], shape[-1]) )
    #
    #    y = K.depthwise_conv2d(x, kernel, strides=(1, 1), padding='valid')
    #
    #    y_shape = K.shape(y)
    #    return y#K.reshape(y, (shape[0], y_shape[1], y_shape[2], y_shape[3] ) )

    image = cv2.imread('D:\\DeepFaceLab\\test\\00000.png').astype(np.float32) / 255.0
    image = cv2.resize ( image, (128,128) )

    image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims (image, -1)
    image_shape = image.shape

    image2 = cv2.imread('D:\\DeepFaceLab\\test\\00001.png').astype(np.float32) / 255.0
    #image2 = cv2.cvtColor (image2, cv2.COLOR_BGR2GRAY)
    #image2 = np.expand_dims (image2, -1)
    image2_shape = image2.shape

    image_tensor = K.placeholder(shape=[ 1, image_shape[0], image_shape[1], image_shape[2] ], dtype="float32" )
    image2_tensor = K.placeholder(shape=[ 1, image_shape[0], image_shape[1], image_shape[2] ], dtype="float32" )

    #loss = reducer(image_tensor)
    #loss = K.reshape (loss, (-1,246,246, 11,11,3) )
    tf = nnlib.tf

    sh = K.int_shape(image_tensor)[1]
    wnd_size = 16
    step_size = 8
    k = (sh-wnd_size) // step_size + 1

    loss = tf.image.extract_image_patches(image_tensor, [1,k,k,1], [1,1,1,1], [1,step_size,step_size,1], 'VALID')
    print(loss)

    f = K.function ( [image_tensor], [loss] )
    x = f ( [ np.expand_dims(image,0) ] )[0][0]

    import code
    code.interact(local=dict(globals(), **locals()))

    for i in range( x.shape[2] ):
        img = x[:,:,i:i+1]

        cv2.imshow('', (img*255).astype(np.uint8) )
        cv2.waitKey(0)

    #for i in range( len(x) ):
    #    for j in range ( len(x) ):
    #        img = x[i,j]
    #        import code
    #        code.interact(local=dict(globals(), **locals()))
    #
    #        cv2.imshow('', (x[i,j]*255).astype(np.uint8) )
    #        cv2.waitKey(0)

    import code
    code.interact(local=dict(globals(), **locals()))


    from nnlib import nnlib
    exec( nnlib.import_all(), locals(), globals() )

    PNet_Input = Input ( (None, None,3) )
    x = PNet_Input
    x = Conv2D (10, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    x = PReLU (shared_axes=[1,2], name="PReLU1" )(x)
    x = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
    x = Conv2D (16, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    x = PReLU (shared_axes=[1,2], name="PReLU2" )(x)
    x = Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
    x = PReLU (shared_axes=[1,2], name="PReLU3" )(x)
    prob = Conv2D (2, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv41")(x)
    prob = Softmax()(prob)
    x = Conv2D (4, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv42")(x)

    PNet_model = Model(PNet_Input, [x,prob] )
    PNet_model.load_weights ( (Path(mtcnn.__file__).parent / 'mtcnn_pnet.h5').__str__() )

    RNet_Input = Input ( (24, 24, 3) )
    x = RNet_Input
    x = Conv2D (28, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    x = PReLU (shared_axes=[1,2], name="prelu1" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='same' ) (x)
    x = Conv2D (48, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    x = PReLU (shared_axes=[1,2], name="prelu2" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='valid' ) (x)
    x = Conv2D (64, kernel_size=(2,2), strides=(1,1), padding='valid', name="conv3")(x)
    x = PReLU (shared_axes=[1,2], name="prelu3" )(x)
    x = Lambda ( lambda x: K.reshape (x, (-1, np.prod(K.int_shape(x)[1:]),) ), output_shape=(np.prod(K.int_shape(x)[1:]),) ) (x)
    x = Dense (128, name='conv4')(x)
    x = PReLU (name="prelu4" )(x)
    prob = Dense (2, name='conv51')(x)
    prob = Softmax()(prob)
    x = Dense (4, name='conv52')(x)
    RNet_model = Model(RNet_Input, [x,prob] )
    RNet_model.load_weights ( (Path(mtcnn.__file__).parent / 'mtcnn_rnet.h5').__str__() )

    ONet_Input = Input ( (48, 48, 3) )
    x = ONet_Input
    x = Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    x = PReLU (shared_axes=[1,2], name="prelu1" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='same' ) (x)
    x = Conv2D (64, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    x = PReLU (shared_axes=[1,2], name="prelu2" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='valid' ) (x)
    x = Conv2D (64, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
    x = PReLU (shared_axes=[1,2], name="prelu3" )(x)
    x = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
    x = Conv2D (128, kernel_size=(2,2), strides=(1,1), padding='valid', name="conv4")(x)
    x = PReLU (shared_axes=[1,2], name="prelu4" )(x)
    x = Lambda ( lambda x: K.reshape (x, (-1, np.prod(K.int_shape(x)[1:]),) ), output_shape=(np.prod(K.int_shape(x)[1:]),) ) (x)
    x = Dense (256, name='conv5')(x)
    x = PReLU (name="prelu5" )(x)
    prob = Dense (2, name='conv61')(x)
    prob = Softmax()(prob)
    x1 = Dense (4, name='conv62')(x)
    x2 = Dense (10, name='conv63')(x)
    ONet_model = Model(ONet_Input, [x1,x2,prob] )
    ONet_model.load_weights ( (Path(mtcnn.__file__).parent / 'mtcnn_onet.h5').__str__() )

    pnet_fun = K.function ( PNet_model.inputs, PNet_model.outputs )
    rnet_fun = K.function ( RNet_model.inputs, RNet_model.outputs )
    onet_fun = K.function ( ONet_model.inputs, ONet_model.outputs )

    pnet_test_data = np.random.uniform ( size=(1, 64,64,3) )
    pnet_result1, pnet_result2 = pnet_fun ([pnet_test_data])

    rnet_test_data = np.random.uniform ( size=(1,24,24,3) )
    rnet_result1, rnet_result2 = rnet_fun ([rnet_test_data])

    onet_test_data = np.random.uniform ( size=(1,48,48,3) )
    onet_result1, onet_result2, onet_result3 = onet_fun ([onet_test_data])

    import code
    code.interact(local=dict(globals(), **locals()))

    from nnlib import nnlib
    #exec( nnlib.import_all( nnlib.device.Config(cpu_only=True) ), locals(), globals() )# nnlib.device.Config(cpu_only=True)
    exec( nnlib.import_all(), locals(), globals() )# nnlib.device.Config(cpu_only=True)

    #det1_Input = Input ( (None, None,3) )
    #x = det1_Input
    #x = Conv2D (10, kernel_size=(3,3), strides=(1,1), padding='valid')(x)
    #
    #import code
    #code.interact(local=dict(globals(), **locals()))

    tf = nnlib.tf
    tf_session = nnlib.tf_sess

    with tf.variable_scope('pnet2'):
        data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
        pnet2 = mtcnn.PNet(tf, {'data':data})
        pnet2.load( (Path(mtcnn.__file__).parent / 'det1.npy').__str__(), tf_session)
    with tf.variable_scope('rnet2'):
        data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
        rnet2 = mtcnn.RNet(tf, {'data':data})
        rnet2.load( (Path(mtcnn.__file__).parent / 'det2.npy').__str__(), tf_session)
    with tf.variable_scope('onet2'):
        data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
        onet2 = mtcnn.ONet(tf, {'data':data})
        onet2.load( (Path(mtcnn.__file__).parent / 'det3.npy').__str__(), tf_session)



    pnet_fun = K.function([pnet2.layers['data']],[pnet2.layers['conv4-2'], pnet2.layers['prob1']])
    rnet_fun = K.function([rnet2.layers['data']],[rnet2.layers['conv5-2'], rnet2.layers['prob1']])
    onet_fun = K.function([onet2.layers['data']],[onet2.layers['conv6-2'], onet2.layers['conv6-3'], onet2.layers['prob1']])

    det1_dict = np.load((Path(mtcnn.__file__).parent / 'det1.npy').__str__(), encoding='latin1').item()
    det2_dict = np.load((Path(mtcnn.__file__).parent / 'det2.npy').__str__(), encoding='latin1').item()
    det3_dict = np.load((Path(mtcnn.__file__).parent / 'det3.npy').__str__(), encoding='latin1').item()

    PNet_Input = Input ( (None, None,3) )
    x = PNet_Input
    x = Conv2D (10, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    x = PReLU (shared_axes=[1,2], name="PReLU1" )(x)
    x = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
    x = Conv2D (16, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    x = PReLU (shared_axes=[1,2], name="PReLU2" )(x)
    x = Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
    x = PReLU (shared_axes=[1,2], name="PReLU3" )(x)
    prob = Conv2D (2, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv41")(x)
    prob = Softmax()(prob)
    x = Conv2D (4, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv42")(x)


    PNet_model = Model(PNet_Input, [x,prob] )

    #PNet_model.load_weights ( (Path(mtcnn.__file__).parent / 'mtcnn_pnet.h5').__str__() )
    PNet_model.get_layer("conv1").set_weights ( [ det1_dict['conv1']['weights'], det1_dict['conv1']['biases'] ] )
    PNet_model.get_layer("PReLU1").set_weights ( [ np.reshape(det1_dict['PReLU1']['alpha'], (1,1,-1)) ] )
    PNet_model.get_layer("conv2").set_weights ( [ det1_dict['conv2']['weights'], det1_dict['conv2']['biases'] ] )
    PNet_model.get_layer("PReLU2").set_weights ( [ np.reshape(det1_dict['PReLU2']['alpha'], (1,1,-1)) ] )
    PNet_model.get_layer("conv3").set_weights ( [ det1_dict['conv3']['weights'], det1_dict['conv3']['biases'] ] )
    PNet_model.get_layer("PReLU3").set_weights ( [ np.reshape(det1_dict['PReLU3']['alpha'], (1,1,-1)) ] )
    PNet_model.get_layer("conv41").set_weights ( [ det1_dict['conv4-1']['weights'], det1_dict['conv4-1']['biases'] ] )
    PNet_model.get_layer("conv42").set_weights ( [ det1_dict['conv4-2']['weights'], det1_dict['conv4-2']['biases'] ] )
    PNet_model.save ( (Path(mtcnn.__file__).parent / 'mtcnn_pnet.h5').__str__() )

    pnet_test_data = np.random.uniform ( size=(1, 64,64,3) )
    pnet_result1, pnet_result2 = pnet_fun ([pnet_test_data])
    pnet2_result1, pnet2_result2 =  K.function ( PNet_model.inputs, PNet_model.outputs ) ([pnet_test_data])

    pnet_diff1 = np.mean ( np.abs(pnet_result1 - pnet2_result1) )
    pnet_diff2 = np.mean ( np.abs(pnet_result2 - pnet2_result2) )
    print ("pnet_diff1 = %f, pnet_diff2 = %f, "  % (pnet_diff1, pnet_diff2) )

    RNet_Input = Input ( (24, 24, 3) )
    x = RNet_Input
    x = Conv2D (28, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    x = PReLU (shared_axes=[1,2], name="prelu1" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='same' ) (x)
    x = Conv2D (48, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    x = PReLU (shared_axes=[1,2], name="prelu2" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='valid' ) (x)
    x = Conv2D (64, kernel_size=(2,2), strides=(1,1), padding='valid', name="conv3")(x)
    x = PReLU (shared_axes=[1,2], name="prelu3" )(x)
    x = Lambda ( lambda x: K.reshape (x, (-1, np.prod(K.int_shape(x)[1:]),) ), output_shape=(np.prod(K.int_shape(x)[1:]),) ) (x)
    x = Dense (128, name='conv4')(x)
    x = PReLU (name="prelu4" )(x)
    prob = Dense (2, name='conv51')(x)
    prob = Softmax()(prob)
    x = Dense (4, name='conv52')(x)

    RNet_model = Model(RNet_Input, [x,prob] )

    #RNet_model.load_weights ( (Path(mtcnn.__file__).parent / 'mtcnn_rnet.h5').__str__() )
    RNet_model.get_layer("conv1").set_weights ( [ det2_dict['conv1']['weights'], det2_dict['conv1']['biases'] ] )
    RNet_model.get_layer("prelu1").set_weights ( [ np.reshape(det2_dict['prelu1']['alpha'], (1,1,-1)) ] )
    RNet_model.get_layer("conv2").set_weights ( [ det2_dict['conv2']['weights'], det2_dict['conv2']['biases'] ] )
    RNet_model.get_layer("prelu2").set_weights ( [ np.reshape(det2_dict['prelu2']['alpha'], (1,1,-1)) ] )
    RNet_model.get_layer("conv3").set_weights ( [ det2_dict['conv3']['weights'], det2_dict['conv3']['biases'] ] )
    RNet_model.get_layer("prelu3").set_weights ( [ np.reshape(det2_dict['prelu3']['alpha'], (1,1,-1)) ] )
    RNet_model.get_layer("conv4").set_weights ( [ det2_dict['conv4']['weights'], det2_dict['conv4']['biases'] ] )
    RNet_model.get_layer("prelu4").set_weights ( [ det2_dict['prelu4']['alpha'] ] )
    RNet_model.get_layer("conv51").set_weights ( [ det2_dict['conv5-1']['weights'], det2_dict['conv5-1']['biases'] ] )
    RNet_model.get_layer("conv52").set_weights ( [ det2_dict['conv5-2']['weights'], det2_dict['conv5-2']['biases'] ] )
    RNet_model.save ( (Path(mtcnn.__file__).parent / 'mtcnn_rnet.h5').__str__() )

    #import code
    #code.interact(local=dict(globals(), **locals()))

    rnet_test_data = np.random.uniform ( size=(1,24,24,3) )
    rnet_result1, rnet_result2 = rnet_fun ([rnet_test_data])
    rnet2_result1, rnet2_result2 =  K.function ( RNet_model.inputs, RNet_model.outputs ) ([rnet_test_data])

    rnet_diff1 = np.mean ( np.abs(rnet_result1 - rnet2_result1) )
    rnet_diff2 = np.mean ( np.abs(rnet_result2 - rnet2_result2) )
    print ("rnet_diff1 = %f, rnet_diff2 = %f, "  % (rnet_diff1, rnet_diff2) )


    #################
    '''
    (self.feed('data') #pylint: disable=no-value-for-parameter, no-member
             .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
             .prelu(name='prelu1')
             .max_pool(3, 3, 2, 2, name='pool1')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
             .prelu(name='prelu2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
             .prelu(name='prelu3')
             .max_pool(2, 2, 2, 2, name='pool3')
             .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
             .prelu(name='prelu4')
             .fc(256, relu=False, name='conv5')
             .prelu(name='prelu5')
             .fc(2, relu=False, name='conv6-1')
             .softmax(1, name='prob1'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5') #pylint: disable=no-value-for-parameter
             .fc(10, relu=False, name='conv6-3'))
    '''
    ONet_Input = Input ( (48, 48, 3) )
    x = ONet_Input
    x = Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    x = PReLU (shared_axes=[1,2], name="prelu1" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='same' ) (x)
    x = Conv2D (64, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    x = PReLU (shared_axes=[1,2], name="prelu2" )(x)
    x = MaxPooling2D( pool_size=(3,3), strides=(2,2), padding='valid' ) (x)
    x = Conv2D (64, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
    x = PReLU (shared_axes=[1,2], name="prelu3" )(x)
    x = MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
    x = Conv2D (128, kernel_size=(2,2), strides=(1,1), padding='valid', name="conv4")(x)
    x = PReLU (shared_axes=[1,2], name="prelu4" )(x)
    x = Lambda ( lambda x: K.reshape (x, (-1, np.prod(K.int_shape(x)[1:]),) ), output_shape=(np.prod(K.int_shape(x)[1:]),) ) (x)
    x = Dense (256, name='conv5')(x)
    x = PReLU (name="prelu5" )(x)
    prob = Dense (2, name='conv61')(x)
    prob = Softmax()(prob)
    x1 = Dense (4, name='conv62')(x)
    x2 = Dense (10, name='conv63')(x)

    ONet_model = Model(ONet_Input, [x1,x2,prob] )

    #ONet_model.load_weights ( (Path(mtcnn.__file__).parent / 'mtcnn_onet.h5').__str__() )
    ONet_model.get_layer("conv1").set_weights ( [ det3_dict['conv1']['weights'], det3_dict['conv1']['biases'] ] )
    ONet_model.get_layer("prelu1").set_weights ( [ np.reshape(det3_dict['prelu1']['alpha'], (1,1,-1)) ] )
    ONet_model.get_layer("conv2").set_weights ( [ det3_dict['conv2']['weights'], det3_dict['conv2']['biases'] ] )
    ONet_model.get_layer("prelu2").set_weights ( [ np.reshape(det3_dict['prelu2']['alpha'], (1,1,-1)) ] )
    ONet_model.get_layer("conv3").set_weights ( [ det3_dict['conv3']['weights'], det3_dict['conv3']['biases'] ] )
    ONet_model.get_layer("prelu3").set_weights ( [ np.reshape(det3_dict['prelu3']['alpha'], (1,1,-1)) ] )
    ONet_model.get_layer("conv4").set_weights ( [ det3_dict['conv4']['weights'], det3_dict['conv4']['biases'] ] )
    ONet_model.get_layer("prelu4").set_weights ( [ np.reshape(det3_dict['prelu4']['alpha'], (1,1,-1)) ] )
    ONet_model.get_layer("conv5").set_weights ( [ det3_dict['conv5']['weights'], det3_dict['conv5']['biases'] ] )
    ONet_model.get_layer("prelu5").set_weights ( [ det3_dict['prelu5']['alpha'] ] )
    ONet_model.get_layer("conv61").set_weights ( [ det3_dict['conv6-1']['weights'], det3_dict['conv6-1']['biases'] ] )
    ONet_model.get_layer("conv62").set_weights ( [ det3_dict['conv6-2']['weights'], det3_dict['conv6-2']['biases'] ] )
    ONet_model.get_layer("conv63").set_weights ( [ det3_dict['conv6-3']['weights'], det3_dict['conv6-3']['biases'] ] )
    ONet_model.save ( (Path(mtcnn.__file__).parent / 'mtcnn_onet.h5').__str__() )

    onet_test_data = np.random.uniform ( size=(1,48,48,3) )
    onet_result1, onet_result2, onet_result3 = onet_fun ([onet_test_data])
    onet2_result1, onet2_result2, onet2_result3 =  K.function ( ONet_model.inputs, ONet_model.outputs ) ([onet_test_data])

    onet_diff1 = np.mean ( np.abs(onet_result1 - onet2_result1) )
    onet_diff2 = np.mean ( np.abs(onet_result2 - onet2_result2) )
    onet_diff3 = np.mean ( np.abs(onet_result3 - onet2_result3) )
    print ("onet_diff1 = %f, onet_diff2 = %f, , onet_diff3 = %f "  % (onet_diff1, onet_diff2, onet_diff3) )


    import code
    code.interact(local=dict(globals(), **locals()))





    import code
    code.interact(local=dict(globals(), **locals()))






    #class MTCNNSoftmax(keras.Layer):
    #
    #    def __init__(self, axis=-1, **kwargs):
    #        super(MTCNNSoftmax, self).__init__(**kwargs)
    #        self.supports_masking = True
    #        self.axis = axis
    #
    #    def call(self, inputs):
    #
    #    def softmax(self, target, axis, name=None):
    #        max_axis = self.tf.reduce_max(target, axis, keepdims=True)
    #        target_exp = self.tf.exp(target-max_axis)
    #        normalize = self.tf.reduce_sum(target_exp, axis, keepdims=True)
    #        softmax = self.tf.div(target_exp, normalize, name)
    #        return softmax
    #        #return activations.softmax(inputs, axis=self.axis)
    #
    #    def get_config(self):
    #        config = {'axis': self.axis}
    #        base_config = super(MTCNNSoftmax, self).get_config()
    #        return dict(list(base_config.items()) + list(config.items()))
    #
    #    def compute_output_shape(self, input_shape):
    #        return input_shape

    from nnlib import nnlib
    exec( nnlib.import_all(), locals(), globals() )




    image = cv2.imread('D:\\DeepFaceLab\\test\\00000.png').astype(np.float32) / 255.0
    image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims (image, -1)
    image_shape = image.shape

    image2 = cv2.imread('D:\\DeepFaceLab\\test\\00001.png').astype(np.float32) / 255.0
    image2 = cv2.cvtColor (image2, cv2.COLOR_BGR2GRAY)
    image2 = np.expand_dims (image2, -1)
    image2_shape = image2.shape

    #cv2.imshow('', image)


    image_tensor = K.placeholder(shape=[ 1, image_shape[0], image_shape[1], image_shape[2] ], dtype="float32" )
    image2_tensor = K.placeholder(shape=[ 1, image_shape[0], image_shape[1], image_shape[2] ], dtype="float32" )

    blurred_image_tensor = gaussian_blur(16.0)(image_tensor)
    x, = nnlib.tf_sess.run ( blurred_image_tensor, feed_dict={image_tensor: np.expand_dims(image,0)} )
    cv2.imshow('', (x*255).astype(np.uint8) )
    cv2.waitKey(0)

    import code
    code.interact(local=dict(globals(), **locals()))


    #os.environ['plaidML'] = '1'
    from nnlib import nnlib

    dvc = nnlib.device.Config(force_gpu_idx=1)
    exec( nnlib.import_all(dvc), locals(), globals() )

    tf = nnlib.tf

    image = cv2.imread('D:\\DeepFaceLab\\test\\00000.png').astype(np.float32) / 255.0
    image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims (image, -1)
    image_shape = image.shape

    image2 = cv2.imread('D:\\DeepFaceLab\\test\\00001.png').astype(np.float32) / 255.0
    image2 = cv2.cvtColor (image2, cv2.COLOR_BGR2GRAY)
    image2 = np.expand_dims (image2, -1)
    image2_shape = image2.shape

    image1_tensor = K.placeholder(shape=[ 1, image_shape[0], image_shape[1], image_shape[2] ], dtype="float32" )
    image2_tensor = K.placeholder(shape=[ 1, image_shape[0], image_shape[1], image_shape[2] ], dtype="float32" )



    #import code
    #code.interact(local=dict(globals(), **locals()))
    def manual_conv(input, filter, strides, padding):
          h_f, w_f, c_in, c_out = filter.get_shape().as_list()
          input_patches = tf.extract_image_patches(input, ksizes=[1, h_f, w_f, 1 ], strides=strides, rates=[1, 1, 1, 1], padding=padding)
          return input_patches
          filters_flat = tf.reshape(filter, shape=[h_f*w_f*c_in, c_out])
          return tf.einsum("ijkl,lm->ijkm", input_patches, filters_flat)

    def extract_image_patches(x, ksizes, ssizes, padding='SAME',
                          data_format='channels_last'):
        """Extract the patches from an image.
        # Arguments
            x: The input image
            ksizes: 2-d tuple with the kernel size
            ssizes: 2-d tuple with the strides size
            padding: 'same' or 'valid'
            data_format: 'channels_last' or 'channels_first'
        # Returns
            The (k_w,k_h) patches extracted
            TF ==> (batch_size,w,h,k_w,k_h,c)
            TH ==> (batch_size,w,h,c,k_w,k_h)
        """
        kernel = [1, ksizes[0], ksizes[1], 1]
        strides = [1, ssizes[0], ssizes[1], 1]
        if data_format == 'channels_first':
            x = K.permute_dimensions(x, (0, 2, 3, 1))
        bs_i, w_i, h_i, ch_i = K.int_shape(x)
        patches = tf.extract_image_patches(x, kernel, strides, [1, 1, 1, 1],
                                           padding)
        # Reshaping to fit Theano
        bs, w, h, ch = K.int_shape(patches)
        reshaped = tf.reshape(patches, [-1, w, h, tf.floordiv(ch, ch_i), ch_i])
        final_shape = [-1, w, h, ch_i, ksizes[0], ksizes[1]]
        patches = tf.reshape(tf.transpose(reshaped, [0, 1, 2, 4, 3]), final_shape)
        if data_format == 'channels_last':
            patches = K.permute_dimensions(patches, [0, 1, 2, 4, 5, 3])
        return patches

    m = 32
    c_in = 3
    c_out = 16

    filter_sizes = [5, 11]
    strides = [1]
    #paddings = ["VALID", "SAME"]

    for fs in filter_sizes:
        h = w = 128
        h_f = w_f = fs
        stri = 2
        #print "Testing for", imsize, fs, stri, pad

        #tf.reset_default_graph()
        X = tf.constant(1.0+np.random.rand(m, h, w, c_in), tf.float32)
        W = tf.constant(np.ones([h_f, w_f, c_in, h_f*w_f*c_in]), tf.float32)


        Z = tf.nn.conv2d(X, W, strides=[1, stri, stri, 1], padding="VALID")
        Z_manual = manual_conv(X, W, strides=[1, stri, stri, 1], padding="VALID")
        Z_2 = extract_image_patches (X, (fs,fs), (stri,stri),  padding="VALID")
        import code
        code.interact(local=dict(globals(), **locals()))
        #
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        Z_, Z_manual_ = sess.run([Z, Z_manual])
        #self.assertEqual(Z_.shape, Z_manual_.shape)
        #self.assertTrue(np.allclose(Z_, Z_manual_, rtol=1e-05))
        sess.close()


        import code
        code.interact(local=dict(globals(), **locals()))





    #k_loss_t = keras_style_loss()(image1_tensor, image2_tensor)
    #k_loss_run = K.function( [image1_tensor, image2_tensor],[k_loss_t])
    #import code
    #code.interact(local=dict(globals(), **locals()))
    #image = np.expand_dims(image,0)
    #image2 = np.expand_dims(image2,0)
    #k_loss = k_loss_run([image, image2])
    #t_loss = t_loss_run([image, image2])




    #x, = tf_sess_run ([np.expand_dims(image,0)])
    #x = x[0]
    ##import code
    ##code.interact(local=dict(globals(), **locals()))



    image = cv2.imread('D:\\DeepFaceLab\\test\\00000.png').astype(np.float32) / 255.0
    image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    image = np.expand_dims (image, -1)
    image_shape = image.shape

    image2 = cv2.imread('D:\\DeepFaceLab\\test\\00001.png').astype(np.float32) / 255.0
    image2 = cv2.cvtColor (image2, cv2.COLOR_BGR2GRAY)
    image2 = np.expand_dims (image2, -1)
    image2_shape = image2.shape

    image_tensor = tf.placeholder(tf.float32, shape=[1, image_shape[0], image_shape[1], image_shape[2] ])
    image2_tensor = tf.placeholder(tf.float32, shape=[1, image2_shape[0], image2_shape[1], image2_shape[2] ])

    blurred_image_tensor = sl(image_tensor, image2_tensor)
    x = tf_sess.run ( blurred_image_tensor, feed_dict={image_tensor: np.expand_dims(image,0), image2_tensor: np.expand_dims(image2,0) } )

    cv2.imshow('', x[0])
    cv2.waitKey(0)
    import code
    code.interact(local=dict(globals(), **locals()))

    while True:
        image = cv2.imread('D:\\DeepFaceLab\\workspace\\data_src\\aligned\\00000.png').astype(np.float32) / 255.0
        image = cv2.resize(image, (256,256))
        image = random_transform( image )
        warped_img, target_img = random_warp( image )

        #cv2.imshow('', image)
        #cv2.waitKey(0)

        cv2.imshow('', warped_img)
        cv2.waitKey(0)
        cv2.imshow('', target_img)
        cv2.waitKey(0)

    import code
    code.interact(local=dict(globals(), **locals()))

    import code
    code.interact(local=dict(globals(), **locals()))

    return


    def keras_gaussian_blur(radius=2.0):
        def gaussian(x, mu, sigma):
            return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

        def make_kernel(sigma):
            kernel_size = max(3, int(2 * 2 * sigma + 1))
            mean = np.floor(0.5 * kernel_size)
            kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
            np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
            kernel = np_kernel / np.sum(np_kernel)
            return kernel

        gauss_kernel = make_kernel(radius)
        gauss_kernel = gauss_kernel[:, :,np.newaxis, np.newaxis]

        #import code
        #code.interact(local=dict(globals(), **locals()))
        def func(input):
            inputs = [ input[:,:,:,i:i+1]  for i in range( K.int_shape( input )[-1] ) ]

            outputs = []
            for i in range(len(inputs)):
                outputs += [ K.conv2d( inputs[i] , K.constant(gauss_kernel) , strides=(1,1), padding="same") ]

            return K.concatenate (outputs, axis=-1)
        return func

    def keras_style_loss(gaussian_blur_radius=0.0, loss_weight=1.0, epsilon=1e-5):
        if gaussian_blur_radius > 0.0:
            gblur = keras_gaussian_blur(gaussian_blur_radius)

        def sd(content, style):
            content_nc = K.int_shape(content)[-1]
            style_nc = K.int_shape(style)[-1]
            if content_nc != style_nc:
                raise Exception("keras_style_loss() content_nc != style_nc")

            axes = [1,2]
            c_mean, c_var = K.mean(content, axis=axes, keepdims=True), K.var(content, axis=axes, keepdims=True)
            s_mean, s_var = K.mean(style, axis=axes, keepdims=True), K.var(style, axis=axes, keepdims=True)
            c_std, s_std = K.sqrt(c_var + epsilon), K.sqrt(s_var + epsilon)

            mean_loss = K.sum(K.square(c_mean-s_mean))
            std_loss = K.sum(K.square(c_std-s_std))

            return (mean_loss + std_loss) * loss_weight

        def func(target, style):
            if gaussian_blur_radius > 0.0:
                return sd( gblur(target), gblur(style))
            else:
                return sd( target, style )
        return func

    data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
    pnet2 = mtcnn.PNet(tf, {'data':data})
    filename = str(Path(mtcnn.__file__).parent/'det1.npy')
    pnet2.load(filename, tf_sess)

    pnet_fun = K.function([pnet2.layers['data']],[pnet2.layers['conv4-2'], pnet2.layers['prob1']])

    import code
    code.interact(local=dict(globals(), **locals()))

    return


    while True:
        img_bgr = np.random.rand ( 268, 640, 3 )
        img_size = img_bgr.shape[1], img_bgr.shape[0]

        mat = np.array( [[ 1.99319629e+00, -1.81504324e-01, -3.62479778e+02],
                         [ 1.81504324e-01,  1.99319629e+00, -8.05396709e+01]] )

        tmp_0 = np.random.rand ( 128,128 ) - 0.1
        tmp   = np.expand_dims (tmp_0, axis=-1)

        mask = np.ones ( tmp.shape, dtype=np.float32)
        mask_border_size = int ( mask.shape[1] * 0.0625 )
        mask[:,0:mask_border_size,:] = 0
        mask[:,-mask_border_size:,:] = 0

        x = cv2.warpAffine( mask, mat, img_size, np.zeros(img_bgr.shape, dtype=np.float32), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4, cv2.BORDER_TRANSPARENT )

        if len ( np.argwhere( np.isnan(x) ) ) == 0:
            print ("fine")
        else:
            print ("wtf")

    import code
    code.interact(local=dict(globals(), **locals()))

    return

    aligned_path_image_paths = Path_utils.get_image_paths("E:\\FakeFaceVideoSources\\Datasets\\CelebA aligned")

    a = []
    r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
    t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

    yaws = []
    pitchs = []
    for filepath in tqdm(aligned_path_image_paths, desc="test", ascii=True ):
        filepath = Path(filepath)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath), print_on_no_embedded_data=True )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath), print_on_no_embedded_data=True )
        else:
            print ("%s is not a dfl image file" % (filepath.name) )

        #source_filename_stem = Path( dflimg.get_source_filename() ).stem
        #if source_filename_stem not in alignments.keys():
        #    alignments[ source_filename_stem ] = []


        #focal_length = dflimg.shape[1]
        #camera_center = (dflimg.shape[1] / 2, dflimg.shape[0] / 2)
        #camera_matrix = np.array(
        #    [[focal_length, 0, camera_center[0]],
        #     [0, focal_length, camera_center[1]],
        #     [0, 0, 1]], dtype=np.float32)
        #
        landmarks = dflimg.get_landmarks()
        #
        #lm = landmarks.astype(np.float32)

        img = cv2_imread (str(filepath)) / 255.0

        img = LandmarksProcessor.draw_landmarks(img, landmarks, (1,1,1) )


        #(_, rotation_vector, translation_vector) = cv2.solvePnP(
        #    LandmarksProcessor.landmarks_68_3D,
        #    lm,
        #    camera_matrix,
        #    np.zeros((4, 1)) )
        #
        #rme = mathlib.rotationMatrixToEulerAngles( cv2.Rodrigues(rotation_vector)[0] )
        #import code
        #code.interact(local=dict(globals(), **locals()))

        #rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)


        #img2 = image_utils.get_text_image ( (256,10, 3), str(rotation_vector) )
        pitch, yaw = LandmarksProcessor.estimate_pitch_yaw (landmarks)
        yaws += [yaw]
        #print(pitch, yaw)
        #cv2.imshow ("", (img * 255).astype(np.uint8) )
        #cv2.waitKey(0)
        #a += [ rotation_vector]
    yaws = np.array(yaws)
    import code
    code.interact(local=dict(globals(), **locals()))






        #alignments[ source_filename_stem ].append (dflimg.get_source_landmarks())
        #alignments.append (dflimg.get_source_landmarks())







    o = np.ones ( (128,128,3), dtype=np.float32 )
    cv2.imwrite ("D:\\temp\\z.jpg", o)

    #DFLJPG.embed_data ("D:\\temp\\z.jpg", )

    dfljpg = DFLJPG.load("D:\\temp\\z.jpg")

    import code
    code.interact(local=dict(globals(), **locals()))

    return



    import sys, numpy; print(numpy.__version__, sys.version)
    sq = multiprocessing.Queue()
    cq = multiprocessing.Queue()

    p = multiprocessing.Process(target=subprocess, args=(sq,cq,))
    p.start()

    while True:
        cq.get() #waiting numpy array
        sq.put (1) #send message we are ready to get more

    #import code
    #code.interact(local=dict(globals(), **locals()))

    os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'

    from nnlib import nnlib
    exec( nnlib.import_all(), locals(), globals() )




    #import tensorflow as tf
    #tf_module = tf
    #
    #config = tf_module.ConfigProto()
    #config.gpu_options.force_gpu_compatible = True
    #tf_session = tf_module.Session(config=config)
    #
    #srgb_tensor = tf.placeholder("float", [None, None, 3])
    #
    #filename = Path(__file__).parent / '00050.png'
    #img = cv2.imread(str(filename)).astype(np.float32) / 255.0
    #
    #lab_tensor = rgb_to_lab (tf_module, srgb_tensor)
    #
    #rgb_tensor = lab_to_rgb (tf_module, lab_tensor)
    #
    #rgb = tf_session.run(rgb_tensor, feed_dict={srgb_tensor: img})
    #cv2.imshow("", rgb)
    #cv2.waitKey(0)

    #from skimage import io, color
    #def_lab = color.rgb2lab(img)
    #
    #t = time.time()
    #def_lab = color.rgb2lab(img)
    #print ( time.time() - t )
    #
    #lab = tf_session.run(lab_tensor, feed_dict={srgb_tensor: img})
    #
    #t = time.time()
    #lab = tf_session.run(lab_tensor, feed_dict={srgb_tensor: img})
    #print ( time.time() - t )






    #lab_clr = color.rgb2lab(img_bgr)
    #lab_bw = color.rgb2lab(out_img)
    #tmp_channel, a_channel, b_channel = cv2.split(lab_clr)
    #l_channel, tmp2_channel, tmp3_channel = cv2.split(lab_bw)
    #img_LAB = cv2.merge((l_channel,a_channel, b_channel))
    #out_img = color.lab2rgb(lab.astype(np.float64))
    #
    #cv2.imshow("", out_img)
    #cv2.waitKey(0)

    #import code
    #code.interact(local=dict(globals(), **locals()))



if __name__ == "__main__":

    #import os
    #os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    #os.environ["PLAIDML_DEVICE_IDS"] = "opencl_nvidia_geforce_gtx_1060_6gb.0"
    #import keras
    #import numpy as np
    #import cv2
    #import time
    #K = keras.backend
    #
    #
    #
    #PNet_Input = keras.layers.Input ( (None, None,3) )
    #x = PNet_Input
    #x = keras.layers.Conv2D (10, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv1")(x)
    #x = keras.layers.PReLU (shared_axes=[1,2], name="PReLU1" )(x)
    #x = keras.layers.MaxPooling2D( pool_size=(2,2), strides=(2,2), padding='same' ) (x)
    #x = keras.layers.Conv2D (16, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv2")(x)
    #x = keras.layers.PReLU (shared_axes=[1,2], name="PReLU2" )(x)
    #x = keras.layers.Conv2D (32, kernel_size=(3,3), strides=(1,1), padding='valid', name="conv3")(x)
    #x = keras.layers.PReLU (shared_axes=[1,2], name="PReLU3" )(x)
    #prob = keras.layers.Conv2D (2, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv41")(x)
    #x = keras.layers.Conv2D (4, kernel_size=(1,1), strides=(1,1), padding='valid', name="conv42")(x)
    #
    #pnet = K.function ([PNet_Input], [x,prob] )
    #
    #img = np.random.uniform ( size=(1920,1920,3) )
    #minsize=80
    #factor=0.95
    #factor_count=0
    #h=img.shape[0]
    #w=img.shape[1]
    #
    #minl=np.amin([h, w])
    #m=12.0/minsize
    #minl=minl*m
    ## create scale pyramid
    #scales=[]
    #while minl>=12:
    #    scales += [m*np.power(factor, factor_count)]
    #    minl = minl*factor
    #    factor_count += 1
    #    # first stage
    #    for scale in scales:
    #        hs=int(np.ceil(h*scale))
    #        ws=int(np.ceil(w*scale))
    #        im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
    #        im_data = (im_data-127.5)*0.0078125
    #        img_x = np.expand_dims(im_data, 0)
    #        img_x = np.transpose(img_x, (0,2,1,3))
    #        t = time.time()
    #        out = pnet([img_x])
    #        t = time.time() - t
    #        print (img_x.shape, t)
    #
    #import code
    #code.interact(local=dict(globals(), **locals()))

    #os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    #os.environ["PLAIDML_DEVICE_IDS"] = "opencl_nvidia_geforce_gtx_1060_6gb.0"
    #import keras
    #K = keras.backend
    #
    #image = np.random.uniform ( size=(1,256,256,3) )
    #image2 = np.random.uniform ( size=(1,256,256,3) )
    #
    #y_true = K.placeholder ( (None,) + image.shape[1:] )
    #y_pred = K.placeholder ( (None,) + image2.shape[1:] )
    #
    #def reducer(x):
    #    shape = K.shape(x)
    #    x = K.reshape(x, (-1, shape[-3] , shape[-2], shape[-1]) )
    #    y = K.depthwise_conv2d(x, K.constant(np.ones( (11,11,3,1) )), strides=(1, 1), padding='valid' )
    #    y_shape = K.shape(y)
    #    return K.reshape(y, (shape[0], y_shape[1], y_shape[2], y_shape[3] ) )
    #
    #mean0 = reducer(y_true)
    #mean1 = reducer(y_pred)
    #luminance = mean0 * mean1
    #cs = y_true * y_pred
    #
    #result = K.function([y_true, y_pred],[luminance, cs]) ([image, image2])
    #
    #print (result)
    #import code
    #code.interact(local=dict(globals(), **locals()))


    main()

sys.exit()



import sys
import numpy as np

def main():
    import tensorflow
    ####import keras
    keras = tensorflow.keras
    K = keras.backend
    KL = keras.layers

    bgr_shape = (64, 64, 3)
    inp = KL.Input(bgr_shape)
    outp = K.mean(inp)

    flow = K.function([inp],[outp])
    flow ( [np.zeros ( (1,64,64,3), dtype=np.float32 )] )

if __name__ == "__main__":
    main()

sys.exit()



import numpy as np
import tensorflow as tf
keras = tf.keras
KL = keras.layers
K = keras.backend

bgr_shape = (128, 128, 3)
#batch_size = 132 #max -TF.keras-tf.1.11.0-cuda 9
batch_size = 90 #max -TF.keras-tf.1.13.1-cuda 10

class PixelShuffler(keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(PixelShuffler, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs):

        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))


        batch_size, h, w, c = input_shape
        if batch_size is None:
            batch_size = -1
        rh, rw = self.size
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)

        out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
        out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
        out = K.reshape(out, (batch_size, oh, ow, oc))
        return out

    def compute_output_shape(self, input_shape):

        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))


        height = input_shape[1] * self.size[0] if input_shape[1] is not None else None
        width = input_shape[2] * self.size[1] if input_shape[2] is not None else None
        channels = input_shape[3] // self.size[0] // self.size[1]

        if channels * self.size[0] * self.size[1] != input_shape[3]:
            raise ValueError('channels of input and size are incompatible')

        return (input_shape[0],
                height,
                width,
                channels)

    def get_config(self):
        config = {'size': self.size}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

def upscale (dim):
    def func(x):
        return PixelShuffler()((KL.Conv2D(dim * 4, kernel_size=3, strides=1, padding='same')(x)))
    return func

inp = KL.Input(bgr_shape)
x = inp
x = KL.Conv2D(128, 5, strides=2, padding='same')(x)
x = KL.Conv2D(256, 5, strides=2, padding='same')(x)
x = KL.Conv2D(512, 5, strides=2, padding='same')(x)
x = KL.Conv2D(1024, 5, strides=2, padding='same')(x)
x = KL.Dense(1024)(KL.Flatten()(x))
x = KL.Dense(8 * 8 * 1024)(x)
x = KL.Reshape((8, 8, 1024))(x)
x = upscale(512)(x)
x = upscale(256)(x)
x = upscale(128)(x)
x = upscale(64)(x)
x = KL.Conv2D(3, 5, strides=1, padding='same')(x)

model = keras.models.Model ([inp], [x])
model.compile(optimizer=keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999), loss='mae')

training_data = np.zeros ( (batch_size,128,128,3) )
loss = model.train_on_batch( [training_data], [training_data] )
print ("FINE")

import code
code.interact(local=dict(globals(), **locals()))

#import keras
#batch_size = 84 #max -keras-tf.1.11.0-cuda 9
#batch_size = 80 #max -keras-tf.1.13.1-cuda 10

#inp = KL.Input(bgr_shape)
#inp2 = KL.Input(bgr_shape)
#outp = model(inp)
#loss = K.mean(K.abs(outp-inp2))
#train_func = K.function ([inp,inp2],[loss], keras.optimizers.Adam(lr=5e-5, beta_1=0.5, beta_2=0.999).get_updates(loss, model.trainable_weights) )
#
#loss = train_func( [training_data, training_data] )


"""
class Adam(keras.optimizers.Optimizer):
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(Adam, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.param_states = {}
            for key, st in kwargs.get('param_states',{}).items():
                nst = {}
                nst['iter'] = K.variable(st['iter'], dtype='int64')
                nst['m'] = K.variable(st['m'], dtype=st['m_dtype'])
                nst['v'] = K.variable(st['v'], dtype=st['v_dtype'])
                if amsgrad:
                    nst['vhat'] = K.variable(st['vhat'], dtype=st['vhat_dtype'])
                self.param_states[key] = nst

            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @keras.legacy.interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        self.updates = []
        self.weights = []
        for p, g in zip(params, grads):
            key = p.name
            st = self.param_states.get(key, None)
            if st is None:
                st = {}
                st['iter'] = K.variable( 0, dtype='int64')
                st['m']    = K.zeros(K.int_shape(p), dtype=K.dtype(p))
                st['v']    = K.zeros(K.int_shape(p), dtype=K.dtype(p))
                st['vhat'] = K.zeros(K.int_shape(p), dtype=K.dtype(p)) \
                             if self.amsgrad else \
                             K.zeros(1, dtype=K.dtype(p))
                self.param_states[key] = st

            iter = st['iter']
            m = st['m']
            v = st['v']
            vhat = st['vhat']
            self.weights += [iter, m, v, vhat]
            self.updates.append(K.update_add(iter, 1))

            t = K.cast(iter, K.floatx()) + 1

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)

            lr = self.lr
            if self.initial_decay > 0:
                lr = lr * (1. / (1. + self.decay * t))

            lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                               (1. - K.pow(self.beta_1, t)))

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                self.updates.append(K.update(vhat, vhat_t))
                denom = K.sqrt(vhat_t) + self.epsilon
            else:
                denom = K.sqrt(v_t) + self.epsilon

            p_t = p - lr_t * m_t / denom

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                p_t = p.constraint(p_t)

            self.updates.append(K.update(p, p_t))
        return self.updates

    def get_config(self):
        param_states = {}
        for key, st in self.param_states.items():
            nst = {}
            nst['iter'] = K.get_value(st['iter'])
            nst['m'] = m = K.get_value(st['m'])
            nst['m_dtype'] = m.dtype.name
            nst['v'] = v = K.get_value(st['v'])
            nst['v_dtype'] = v.dtype.name
            nst['vhat'] = vhat = K.get_value(st['vhat'])
            nst['vhat_dtype'] = vhat.dtype.name
            param_states[key] = nst

        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'param_states': param_states
                  }
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        """



"""
class KFunctionAdam(object):
    def __init__ (self, feed_list, output_list, loss, params, lr=0.001, beta_1=0.9, beta_2=0.999, cpu_mode=0):
        #K.backend == 'tensorflow'

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')

        if cpu_mode==0:
            self.ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            self.vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            self.ms = [np.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
            self.vs = [np.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]


        self.epsilon = K.epsilon()
        self.cpu_mode = cpu_mode

        self.params = params
        self.grad = K.gradients(loss, params)
        self.f_output = K.function(feed_list, output_list)

        self.f_grad = K.function(feed_list, self.grad)
"""
