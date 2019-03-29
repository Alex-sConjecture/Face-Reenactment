#import FaceLandmarksExtractor

import os
import numpy as np
import dlib
import torch

if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    os.environ.pop('CUDA_VISIBLE_DEVICES')

os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2'
import tensorflow as tf
config = tf.ConfigProto(device_count={'GPU': 0})

tf_sess = tf.Session(config=config)


import keras

from keras import backend as K
from keras import layers as KL
keras.backend.set_session(tf_sess)
#keras.backend.set_floatx('float16')

import math

import time
import code
import cv2

def t2kw_conv2d (src):
    if src.bias is not None:
        return [ np.moveaxis(src.weight.data.cpu().numpy(), [0,1,2,3], [3,2,0,1]), src.bias.data.cpu().numpy() ]
    else:
        return [ np.moveaxis(src.weight.data.cpu().numpy(), [0,1,2,3], [3,2,0,1])]


def t2kw_bn2d(src):
    return [ src.weight.data.cpu().numpy(), src.bias.data.cpu().numpy(), src.running_mean.cpu().numpy(), src.running_var.cpu().numpy() ]



import face_alignment
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,enable_cuda=False,enable_cudnn=False,use_cnn_face_detector=True).face_alignemnt_net
fa.eval()

def KerasConvBlock(in_planes, out_planes, input, srctorch):
    out1 = KL.BatchNormalization(momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.bn1) )(input)
    out1 = KL.Activation( keras.backend.relu ) (out1)
    out1 = KL.ZeroPadding2D(padding=(1, 1))(out1)
    out1 = KL.Conv2D( int(out_planes/2), kernel_size=3, strides=1, padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.conv1) ) (out1)

    out2 = KL.BatchNormalization(momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.bn2) )(out1)
    out2 = KL.Activation( keras.backend.relu ) (out2)
    out2 = KL.ZeroPadding2D(padding=(1, 1))(out2)
    out2 = KL.Conv2D( int(out_planes/4), kernel_size=3, strides=1, padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.conv2) ) (out2)

    out3 = KL.BatchNormalization(momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.bn3) )(out2)
    out3 = KL.Activation( keras.backend.relu ) (out3)
    out3 = KL.ZeroPadding2D(padding=(1, 1))(out3)
    out3 = KL.Conv2D( int(out_planes/4), kernel_size=3, strides=1, padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.conv3) ) (out3)

    out3 = KL.Concatenate()([out1, out2, out3])

    if in_planes != out_planes:
        downsample = KL.BatchNormalization(momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(srctorch.downsample[0]) )(input)
        downsample = KL.Activation( keras.backend.relu ) (downsample)
        downsample = KL.Conv2D( out_planes, kernel_size=1, strides=1, padding='valid', use_bias = False, weights=t2kw_conv2d(srctorch.downsample[2]) ) (downsample)
        out3 = KL.add ( [out3, downsample] )
    else:
        out3 = KL.add ( [out3, input] )


    return out3

def KerasHourGlass (depth, input, srctorch):

    up1 = KerasConvBlock(256, 256, input, srctorch._modules['b1_%d' % (depth)])

    low1 = KL.AveragePooling2D (pool_size=2, strides=2, padding='valid' )(input)
    low1 = KerasConvBlock (256, 256, low1, srctorch._modules['b2_%d' % (depth)])

    if depth > 1:
        low2 = KerasHourGlass (depth-1, low1, srctorch)
    else:
        low2 = KerasConvBlock(256, 256, low1, srctorch._modules['b2_plus_%d' % (depth)])

    low3 = KerasConvBlock(256, 256, low2, srctorch._modules['b3_%d' % (depth)])

    up2 = KL.UpSampling2D(size=2) (low3)
    return KL.add ( [up1, up2] )

model_path = os.path.join( r'D:\DFLBuild\test\2DFAN-4.h5' )
model_light_path = os.path.join( r'D:\DFLBuild\test\2DFAN-4_light.h5' )
if os.path.exists (model_path):
    t = time.time()
    model = keras.models.load_model (model_path)
    print ('load takes = %f' %( time.time() - t ) )
else:
    _input = keras.layers.Input ( shape=(256,256,3), dtype=K.floatx() )
    x = _input
    x = KL.Lambda ( lambda x: x / 255.0, output_shape=(256,256,3) ) (x)
    x = KL.ZeroPadding2D(padding=(3, 3))(x)
    x = KL.Conv2D( 64, kernel_size=7, strides=2, padding='valid', weights=t2kw_conv2d(fa.conv1) ) (x)

    x = KL.BatchNormalization(momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d(fa.bn1) )(x)
    x = KL.Activation( keras.backend.relu ) (x)

    x = KerasConvBlock (64, 128, x, fa.conv2)
    x = KL.AveragePooling2D (pool_size=2, strides=2, padding='valid' ) (x)
    x = KerasConvBlock (128, 128, x, fa.conv3)
    x = KerasConvBlock (128, 256, x, fa.conv4)

    outputs = []
    previous = x
    for i in range(4):
        ll = KerasHourGlass (4, previous, fa._modules['m%d' % (i) ])
        ll = KerasConvBlock (256,256, ll, fa._modules['top_m_%d' % (i)])

        ll = KL.Conv2D(256, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d( fa._modules['conv_last%d' % (i)] ) ) (ll)
        ll = KL.BatchNormalization(momentum=0.1, epsilon=1e-05, weights=t2kw_bn2d( fa._modules['bn_end%d' % (i)] ) )(ll)
        ll = KL.Activation( keras.backend.relu ) (ll)

        tmp_out = KL.Conv2D(68, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d( fa._modules['l%d' % (i)] ) ) (ll)
        outputs.append(tmp_out)

        if i < 4 - 1:
            ll = KL.Conv2D(256, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d( fa._modules['bl%d' % (i)] ) ) (ll)
            previous = KL.add ( [previous, ll, KL.Conv2D(256, kernel_size=1, strides=1, padding='valid', weights=t2kw_conv2d( fa._modules['al%d' % (i)] ) ) (tmp_out) ] )

    model = keras.models.Model (_input, outputs)
    model.compile ( loss='mse', optimizer='adam' )
    model.save (model_path)

    model_short = keras.models.Model (_input, outputs[-1])
    model_short.compile ( loss='mse', optimizer='adam' )
    model_short.save (model_light_path)

def transform(point, center, scale, resolution, invert=False):
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

def get_preds_fromhm(hm, center=None, scale=None):
    max, idx = torch.max(  hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), True)

    return preds, preds_orig

def transform2(point, center, scale, resolution):
    pt = np.array ( [point[0], point[1], 1.0] )
    h = 200.0 * scale
    m = np.eye(3)
    m[0,0] = resolution / h
    m[1,1] = resolution / h
    m[0,2] = resolution * ( -center[0] / h + 0.5 )
    m[1,2] = resolution * ( -center[1] / h + 0.5 )
    m = np.linalg.inv(m)
    return np.matmul (m, pt)[0:2]

def get_preds_fromhm2(a, center=None, scale=None):
    b = a.reshape ( (a.shape[0], a.shape[1]*a.shape[2]) )
    c = b.argmax(1).reshape ( (a.shape[0], 1) ).repeat(2, axis=1).astype(np.float)
    c[:,0] %= a.shape[2]
    c[:,1] = np.apply_along_axis ( lambda x: np.floor(x / a.shape[2]), 0, c[:,1] )
    for i in range(a.shape[0]):
        pX, pY = int(c[i,0]), int(c[i,1])
        if pX > 0 and pX < 63 and pY > 0 and pY < 63:
            diff = np.array ( [a[i,pY,pX+1]-a[i,pY,pX-1], a[i,pY+1,pX]-a[i,pY-1,pX]] )
            c[i] += np.sign(diff)*0.25

    c += 0.5
    return [ transform2 (c[i], center, scale, a.shape[2]) for i in range(a.shape[0]) ]

#rnd_data = np.random.randint (256, size=(256,256,3) ).astype(np.float32)
rnd_data = cv2.imread ( r"D:\DFLBuild\test\00000.jpg" ).astype(np.float32)

#rnd_data = np.random.random_integers (2, size=(3, 256,256)).astype(np.float32)
#rnd_data = np.array ( [[[1]*256]*256]*3 , dtype=np.float32 )
input_data = np.expand_dims (rnd_data,0)


fa_out_tensor = fa( torch.autograd.Variable( torch.from_numpy(input_data.transpose(0,3,1,2) / 255.0 ), volatile=True) )[-1].data.cpu()
fa_out = fa_out_tensor.numpy()


t = time.time()
m_out = model.predict ( input_data.astype(np.float16) )[-1]
m_out = m_out.transpose(0,3,1,2)
print ('predict takes = %f' %( time.time() - t ) )
t = time.time()

#fa_base_out = fa_base(torch.autograd.Variable( torch.from_numpy(input_data), volatile=True))[0].data.cpu().numpy()

print ( 'shapes = %s , %s , equal == %s ' % (fa_out.shape, m_out.shape, (fa_out.shape == m_out.shape) ) )
print ( 'allclose == %s' %  ( np.allclose(fa_out, m_out) ) ) #all close false but they really close
print ( 'total abs diff outputs = %f' % ( np.sum ( np.abs(np.ndarray.flatten(fa_out-m_out))) ))

###
d = dlib.rectangle(0,0,255,255)

center = torch.FloatTensor(
                    [d.right() - (d.right() - d.left()) / 2.0, d.bottom() -
                     (d.bottom() - d.top()) / 2.0])
center[1] = center[1] - (d.bottom() - d.top()) * 0.12
scale = (d.right() - d.left() + d.bottom() - d.top()) / 195.0
pts, pts_img = get_preds_fromhm (fa_out_tensor, center, scale)
pts_img = pts_img.view(68, 2).numpy()

###
m_pts_img = get_preds_fromhm2 (m_out[0], center, scale)
m_pts_img = np.array( [ ( int(pt[0]), int(pt[1]) ) for pt in m_pts_img ]   )
print ('pts1 == pts2 == %s' % ( np.array_equal(pts_img, m_pts_img) ) )
import code
code.interact(local=dict(globals(), **locals()))

#print ( np.array_equal (fa_out, m_out) ) #>>> False
#code.interact(local=dict(globals(), **locals()))

#code.interact(local=locals())
