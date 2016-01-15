#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
## see lib/train, which is fast_rcnn.train, and there is train_net() function
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys


"""
default parameters for training the net
--gpu  set the gpu id
--solver set the solver file, which is solver.prototxt
--iter set the max iteration times
--weights     set the pre-trained models(<---- based on this model)
--imdb    dataset to be trained on
"""
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


"""
Here is an example that uses to train the VOC2007(use GPU2, with solver and predefined imagenet models), just like finetuned on pretrained models

./tools/train_net.py --gpu 2 --solver models/VGG16/solver.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel 
"""
if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)

    ## get the dataset, here they use voc2007_trainval, which is in the folder of $/data/select_serarch_data/voc_2007_trainval.mat
    imdb = get_imdb(args.imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    
    ##    voc_2007_trainval.mat: 
    ##            this is the data format to be trained on, and it contains 1) boxes(1x5011 cell), 2) images (5011x1 cell)
    ##            for boxes, each one(total is 5011 images) has many bounding-box, for example, boxes[0]= 2443x4, 4 dimensions with 2443 proposals
    ##            for images, there are 5011 images, each one is the image name, image[0] = 000005, means the image name
    
    
    ## get the region of interest area, that will be used for training(NOT ALL the selective_search data is used for training, select some)
    ## this will use the lib/fast_rcnn/train.py's get_training_roidb() to calculate the roidb
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    ## this will call the lib/fast_rcnn/train.py, in it the train() function will use these parameters
    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
