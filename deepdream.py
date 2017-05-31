#!/usr/bin/env python3

import pickle
import lasagne
import theano
import random
from collections import OrderedDict
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.layers import InputLayer
from lasagne.layers import SpatialPyramidPoolingLayer
from lasagne.nonlinearities import softmax, linear
import numpy as np
from theano import tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano.tensor.raw_random import RandomStateType, random_integers
from PIL import Image
import scipy.ndimage as nd
from skimage import exposure
from skimage.util import img_as_float
from matplotlib import pyplot as plt
from collections import deque

#channel-wise means of images from dataset VGG-16 was trained on
VGG16_MEANS = {
    "b": 103.939,
    "g": 116.779,
    "r": 123.68
}

#channels, height, width
VGG16_SHAPE = (3, 224, 224)

def set_layer_as_immutable(layer):
    """Sets layer parameters so as not to be modified in training steps."""
    for k in layer.params.keys():
        layer.params[k] -= {"regularizable", "trainable"}

def mse(pred, tgt):
    """Theano expression: mean-squared-error."""
    return T.square(pred - tgt).mean()

def norm_mse(pred, tgt, alpha):
    """Theano expression: normalized mean-squared-error."""
    return T.square((pred/pred.max() - tgt)/(alpha - tgt)).mean()

def shift(var, sx, sy):
    """Theano expression: rolls tensor in x and y directions."""
    #return T.roll(T.roll(var, sx, -1), sy, -2)
    return np.roll(np.roll(var, sx, -1), sy, -2)

def pre_process(img):
    """Processes input image for network.
        Assumes input image comes:
            -in RGB.
            -in shape (height, width, channels).
            -in the range [0, 255].
    """
    #setting datatype
    img = img.astype("float32")
    #switching from rgb to bgr
    img = img[:, :, ::-1]
    #converting shape from (h, w, c) to (c, h, w)
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    #subtracting each channel by the dataset mean
    for ch_num, ch_name in enumerate("bgr"):
        img[ch_num, :, :] -= VGG16_MEANS[ch_name]

    return img

def de_process(img):
    """Processes image produced by net to original format.
        Assumes input image comes:
            -in BGR.
            -in shape (channels, height, width).
            -in the range [0, 255].
            -in type float.
    """
    img = img.copy()
    #adding dataset mean for each channel
    for ch_num, ch_name in enumerate("bgr"):
        img[ch_num, :, :] += VGG16_MEANS[ch_name]
    #converting shape from (c, h, w) to (h, w, c)
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    #switching from bgr to rgb
    img = img[:, :, ::-1]

    return img

def show_img(img, adjust_contrast=False):
    """Assumes image comes:
        -in RGB.
        -in shape (height, width, channels).
        -in type float.
    """
    #contrast adjusting
    if adjust_contrast:
        img = img*(255.0/np.percentile(img, 99.98))
    #converting to uint8
    img = np.clip(img, 0, 255).astype("uint8")
    #displaying image
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def load_img(filepath):
    """
    Loads image in RGB format from filepath.
    """
    img = np.asarray(Image.open(filepath).convert("RGB"))
    return img

def save_img(img, filepath, adjust_contrast=False):
    """Saves image to filepath.
        Assumes image comes:
            -in RGB.
            -in shape (height, width, channels).
    """
    #contrast adjusting
    if adjust_contrast:
        img = img*(255.0/np.percentile(img, 99.98))
        #print("opa:",
        #    img.shape, img.dtype, img.min(), img.max(), img.mean(), img.std())
        #img = img/img.max()
        #print("opa1.5:",
        #    img.shape, img.dtype, img.min(), img.max(), img.mean(), img.std())
        #img = exposure.equalize_adapthist(img)
        #print("opa2:",
        #    img.shape, img.dtype, img.min(), img.max(), img.mean(), img.std())
        #img = 255*(img.astype("float32"))
    #converting to uint8
    img = np.clip(img, 0, 255).astype("uint8")
    #saving
    img = Image.fromarray(img, mode="RGB")
    img.save(filepath)

def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayer(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}

class Model:
    INPUT_SHAPE = (3, 224, 224)
    OUTPUT_SHAPE = tuple()

    def __init__(self, load_net_from=None):
        #converting to shared var if needed
        """if isinstance(input_var, np.ndarray):
            if len(input_var.shape) < 4:
                input_var = input_var.reshape((1, ) + input_var.shape)
            self.input_var = theano.shared(input_var, name="input_var")
        else:
            self.input_var = input_var"""

        img = T.tensor4(dtype="floatX")

        #the network lasagne model
        self.net = self.get_net_model(img)
        if load_net_from is not None:
            self.load_net(load_net_from)

        pred = lasagne.layers.get_output(self.net["inception_4c/output"],
        #pred = lasagne.layers.get_output(self.net["inception_3b/5x5_reduce"],
        #pred = lasagne.layers.get_output(self.net["output"],
            deterministic=True)
        #print(shp)
        #    deterministic=True)

        #x = np.zeros(shape=(1000,), dtype="float32")
        #for i in [153, 200, 229, 235, 238, 239, 245, 248, 251, 252, 254]:
        #    x[i] = 1.0
        #x2 = theano.shared(x)
        #loss = -T.square(pred - x2).mean()
        loss = T.square(pred).mean()
        #loss = pred.mean()

        g = T.grad(loss, wrt=img)

        #rng = T.shared_randomstreams.RandomStreams(seed=42)

        lr = T.scalar(dtype="floatX")
        #jit = T.iscalar()

        #shift jitter
        #sx = rng.random_integers(low=-jit, high=jit+1)
        #sy = rng.random_integers(low=-jit, high=jit+1)
        #shifting
        #new_img = shift(img, sx, sy)
        #updating var
        #new_img += g*(lr/T.mean(abs(g)))
        new_img = img + g*(lr/T.mean(abs(g)))
        #unshifting
        #new_img = shift(new_img, -sx, -sy)

        #image update
        self._make_pass = theano.function(
            inputs=[img, lr], outputs=[new_img],
        )

        #mean absolute error
        print("DONE")

    def get_net_model(self, input_var=None, inp_shp=None):
        """
        Builds network.
        """
        net = {}

        #net['input'] = InputLayer((None, 3, None, None))
        net['input'] = InputLayer((None, 3, None, None), input_var=input_var)

        net['conv1/7x7_s2'] = ConvLayer(
            net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
        net['pool1/3x3_s2'] = PoolLayer(
            net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
        net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
        net['conv2/3x3_reduce'] = ConvLayer(
            net['pool1/norm1'], 64, 1, flip_filters=False)
        net['conv2/3x3'] = ConvLayer(
            net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
        net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
        net['pool2/3x3_s2'] = PoolLayer(
          net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

        net.update(build_inception_module('inception_3a',
                                          net['pool2/3x3_s2'],
                                          [32, 64, 96, 128, 16, 32]))
        net.update(build_inception_module('inception_3b',
                                          net['inception_3a/output'],
                                          [64, 128, 128, 192, 32, 96]))

        net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'],
                pool_size=3, stride=2, ignore_border=False)

        net.update(build_inception_module('inception_4a',
                                          net['pool3/3x3_s2'],
                                          [64, 192, 96, 208, 16, 48]))
        net.update(build_inception_module('inception_4b',
                                          net['inception_4a/output'],
                                          [64, 160, 112, 224, 24, 64]))
        net.update(build_inception_module('inception_4c',
                                          net['inception_4b/output'],
                                          [64, 128, 128, 256, 24, 64]))
        net.update(build_inception_module('inception_4d',
                                          net['inception_4c/output'],
                                          [64, 112, 144, 288, 32, 64]))
        net.update(build_inception_module('inception_4e',
                                          net['inception_4d/output'],
                                          [128, 256, 160, 320, 32, 128]))
        net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'],
            pool_size=3, stride=2, ignore_border=False)

        net.update(build_inception_module('inception_5a',
                                          net['pool4/3x3_s2'],
                                          [128, 256, 160, 320, 32, 128]))
        net.update(build_inception_module('inception_5b',
                                          net['inception_5a/output'],
                                          [128, 384, 192, 384, 48, 128]))

        net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
        net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                             num_units=1000,
                                             nonlinearity=linear)
        net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                        nonlinearity=softmax)
        net["output"] = net["prob"]

        #for k in net.keys():
            #print("{} = {}:".format(k, net[k]))
        #    if not (k.startswith("fc") or k == "prob"):
        #        continue
        #    for p in net[k].params.keys():
                #print("\t{}: {}".format(p, net[k].params[p]))
        #        print("\tsetting {} as immutable".format(k))
        #        set_layer_as_immutable(net[k])

        return net


    def make_pass(self, img, learning_rate, jitter_shift, clip=False):
        img = img.copy()

        sx = random.randint(-jitter_shift, jitter_shift+1)
        sy = random.randint(-jitter_shift, jitter_shift+1)
        img = shift(img, sx, sy)

        img = img.reshape((1, ) + img.shape)
        img = self._make_pass(img, learning_rate)[0]
        img = img.reshape(img.shape[1:])

        img = shift(img, -sx, -sy)

        if clip:
            #u = new_img.mean()
            img = np.clip(img, -255, 255)
            return img
        else:
            return img

    #def deep_dream(self, img, n_iters=10, n_octaves=4, octave_scale=1.4,
    #        end="output", clip=True, **step_params):
    def deep_dream(self,
            img, n_iters=10, n_octaves=4, octave_scale=1.4, **step_params):

        img = pre_process(img)
        #self.input_var.set_value(img.reshape((1, ) + img.shape))

        #prepare base images for all octaves
        octaves = deque([img])
        for i in range(n_octaves-1):
            octaves.appendleft(nd.zoom(octaves[0],
                (1, 1.0/octave_scale, 1.0/octave_scale), order=1))

        #allocate image for network-produced details
        detail = np.zeros_like(octaves[0])

        for octave, octave_base in enumerate(octaves):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                #upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1, 1.0*w/w1), order=1)

            result = octave_base + detail
            print("oc", octave, "result shape:", result.shape)

            for i in range(n_iters):
                result = self.make_pass(result, **step_params)
                #result = self.make_pass(result, 1.5, 32, clip=False)
                print("iter", i, "result shape:", result.shape,
                    "min, max, mean, std:",
                    result.min(), result.max(), result.mean(), result.std())
                #visualization
                vis = de_process(result)
                #print("vis shape:", vis.shape)

                #show_img(vis)
                save_img(vis, "dream_octave{}_iter{}.jpg".format(octave, i),
                    adjust_contrast=True)
                print(octave, i, vis.shape)

            #extract details produced on the current octave
            detail = result - octave_base

        #returning the resulting image
        return de_process(result)

    def save_net(self, filepath):
        """
        Saves net weights.
        """
        np.savez(filepath, *lasagne.layers.get_all_param_values(
            self.net["output"]))

    def load_net(self, filepath):
        """
        Loads net weights.
        """
        print("loading weights and setting params...", end=" ", flush=True)
        with open(filepath, "rb") as f:
            params = pickle.load(f, encoding="latin1")
        #params["param values"] = params["param values"][:26]
        #del params["param values"][26]
        #del params["param values"][27]
        print(type(params["param values"]))
        for i, p in enumerate(params["param values"]):
            print(i, type(p), p.shape)
        #exit()
        lasagne.layers.set_all_param_values(self.net["output"],
            params["param values"])
        print("done.")

def main():
    #img_fp = "/home/erik/proj/att/att/deep/sky.jpg"
    img_fp = "/home/erik/proj/att/att/deep/head.jpg"
    #img_fp = "/home/erik/proj/att/att/deep/config/wow/neg/17_0.jpg"
    weights_fp = "/home/erik/data/blvc_googlenet.pkl"

    print("loading net...", flush=True)
    model = Model(weights_fp)

    print("opening img...")
    img = load_img(img_fp)
    #print("displaying img...")
    #show_img(img)

    print("deepdreaming...", flush=True)
    model.deep_dream(img, n_octaves=4, n_iters=40, octave_scale=1.4,
        learning_rate=2.0, jitter_shift=32, clip=True)

if __name__ == "__main__":
    main()
