#!/usr/bin/env python3

import lasagne

import theano
from theano import tensor as T

import scipy.ndimage as nd
import numpy as np
import random

import nets
import sys
from collections import deque
from PIL import Image
from matplotlib import pyplot as plt

#network name, see nets module for supported nets
NET_NAME = "googlenet"
#layer to maximize output
NET_LAYER_OUTPUT = "inception_4c/output"
#weights for chosen net
NET_WEIGHTS_FILEPATH = "/home/erik/data/blvc_googlenet.pkl"

#default number of octaves
N_OCTAVES = 4
#default number of iterations per octave
N_ITERS = 10
#default octave scale
OCTAVE_SCALE = 1.4
#default learning rate
LEARNING_RATE = 2.0
#default jitter shift
JITTER_SHIFT = 40
#to clip or not to clip?
CLIP = True
#save intermediate steps
SAVE_STEPS = False
#show intermediate steps
SHOW_STEPS = False

def shift(var, sx, sy):
    """
    Shifts image.
    """
    return np.roll(np.roll(var, sx, -1), sy, -2)

def pre_process(img, dataset_means):
    """
    Processes input image for network.
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
        img[ch_num, :, :] -= dataset_means[ch_name]

    return img

def de_process(img, dataset_means):
    """
    Processes image produced by net to original format.
    Assumes input image comes:
        -in BGR.
        -in shape (channels, height, width).
        -in the range [0, 255].
        -in type float.
    """
    img = img.copy()
    #adding dataset mean for each channel
    for ch_num, ch_name in enumerate("bgr"):
        img[ch_num, :, :] += dataset_means[ch_name]
    #converting shape from (c, h, w) to (h, w, c)
    img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    #switching from bgr to rgb
    img = img[:, :, ::-1]

    return img

def show_img(img, adjust_contrast=False):
    """
    Assumes image comes:
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
    """
    Saves image to filepath.
    Assumes image comes:
        -in RGB.
        -in shape (height, width, channels).
    """
    #contrast adjusting
    if adjust_contrast:
        img = img*(255.0/np.percentile(img, 99.98))
    #converting to uint8
    img = np.clip(img, 0, 255).astype("uint8")
    #saving
    img = Image.fromarray(img, mode="RGB")
    img.save(filepath)

def l2(pred):
    """
    L2 norm.
    """
    return T.square(pred).mean()

class DeepDream:
    def __init__(self,
        net_name="googlenet",
        load_net_from=None,
        output_layer="inception_4c/output",
        objective=l2):

        self.net_name = net_name

        #symbolic var representing image
        img = T.tensor4(dtype="floatX")

        #building net
        net = nets.build_net(net_name, img)
        if load_net_from is None:
            raise ValueError("you must specify net weights file!")
        #loading net weights
        nets.load_net(net, load_net_from)

        #prediction symbolic expression
        pred = lasagne.layers.get_output(net[output_layer],
            deterministic=True)

        #loss function
        obj = objective(pred)
        #gradient of objective
        g = T.grad(obj, wrt=img)
        #symbolic variable for learning rate
        learning_rate = T.scalar(dtype="floatX")
        #normalized gradient ascent step
        new_img = img + g*(learning_rate/T.mean(abs(g)))

        #image update function compilation
        self._make_pass = theano.function(
            inputs=[img, learning_rate], outputs=[new_img],
        )

    def make_pass(self, img, learning_rate, jitter_shift, clip=False):
        """
        Wrapper for _make_pass, applies random jitter shift and maybe clips.
        """
        img = img.copy()

        #applying random shift
        sx = random.randint(-jitter_shift, jitter_shift+1)
        sy = random.randint(-jitter_shift, jitter_shift+1)
        img = shift(img, sx, sy)

        #converting to tensor shape, applying _make_pass, converting back
        img = img.reshape((1, ) + img.shape)
        img = self._make_pass(img, learning_rate)[0]
        img = img.reshape(img.shape[1:])

        #de-shifting
        img = shift(img, -sx, -sy)

        #clipping if needed
        if clip:
            return np.clip(img, -255, 255)
        else:
            return img

    def dream(self, img,
            n_iters=10, n_octaves=4, octave_scale=1.4,
            save_steps=False, show_steps=False,
            **step_params):
        """
        Dreaming loop.
        """

        #pre-processing img
        img = pre_process(img, nets.NETS_MEANS[self.net_name])

        #prepare base images for all octaves
        octaves = deque([img])
        for i in range(n_octaves-1):
            octaves.appendleft(nd.zoom(octaves[0],
                (1, 1.0/octave_scale, 1.0/octave_scale), order=1))

        #allocate image for network-produced details
        detail = np.zeros_like(octaves[0])

        #main dream loop
        for octave, octave_base in enumerate(octaves):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                #upscaling details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1, 1.0*w/w1), order=1)

            result = octave_base + detail

            for i in range(n_iters):
                print("in octave #{}, iteration #{}...".format(octave, i))

                #dream step
                result = self.make_pass(result, **step_params)
                #print("iter", i, "result shape:", result.shape,
                #    "min, max, mean, std:",
                #    result.min(), result.max(), result.mean(), result.std())
                #visualization
                vis_img = de_process(result, nets.NETS_MEANS[self.net_name])

                #displaying/saving if needed
                if save_steps:
                    save_img(vis_img,
                        "dream_octave{}_iter{}.jpg".format(octave, i),
                        adjust_contrast=True)
                if show_steps:
                    show_img(vis_img, adjust_contrast=True)

            #extracting details produced on the current octave
            detail = result - octave_base

        return de_process(result, nets.NETS_MEANS[self.net_name])

def print_usage_and_exit():
    print("usage: {} <image_filepath>".format(sys.argv[0]))
    print("(advanced usage: {} <image_filepath> <n_octaves> <n_iters> "
        "<octave_scale> <learning_rate>".format(sys.argv[0]))
    exit()

def main():
    if len(sys.argv) < 2:
        print_usage_and_exit()
    img_fp = sys.argv[1]

    #setting parameters
    n_octaves = int(sys.argv[2]) if len(sys.argv) > 2 else N_OCTAVES
    n_iters = int(sys.argv[3]) if len(sys.argv) > 3 else N_ITERS
    octave_scale = float(sys.argv[4]) if len(sys.argv) > 4 else OCTAVE_SCALE
    learning_rate = float(sys.argv[5]) if len(sys.argv) > 5 else LEARNING_RATE

    print("building and loading net...", flush=True, end=" ")
    model = DeepDream(NET_NAME, NET_WEIGHTS_FILEPATH, NET_LAYER_OUTPUT)
    print("done")

    print("opening img...", flush=True, end=" ")
    img = load_img(img_fp)
    print("done")

    print("deepdreaming...", flush=True, end=" ")
    result = model.dream(img,
        n_octaves=n_octaves, n_iters=n_iters, octave_scale=octave_scale,
        learning_rate=learning_rate, jitter_shift=JITTER_SHIFT, clip=CLIP,
        save_steps=SAVE_STEPS, show_steps=SHOW_STEPS)
    print("done.")

    result_fp = "{}_dream.jpg".format(img_fp.split(".")[0])
    print("saving result to '{}'...".format(result_fp), flush=True, end=" ")
    save_img(result, result_fp, True)
    print("done.")

if __name__ == "__main__":
    main()
