from __future__ import print_function
import sys
import os
import pickle
import json
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import cv2
import pprint
from torch.autograd import Variable
from data import VOCroot,COCOroot 
from data import AnnotationTransform, COCODetection, VOCDetection, DummyDetection, BaseTransform, VOC_300,VOC_512,COCO_300,COCO_512, COCO_mobile_300

import torch.utils.data as data
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='300',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to test model')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_set', default='', type=str,
                    help='Location of testset')
parser.add_argument('--test_prefix', default='', type=str,
                    help='Prefix of testset')
parser.add_argument('--thresh', default=0.7, type=float,
                    help='Threshold')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
    num_classes = 21
elif args.dataset == 'COCO':
    cfg = (COCO_300, COCO_512)[args.size == '512']
    num_classes = 81
elif args.dataset == 'juggdet':
    cfg = (COCO_300, COCO_512)[args.size == '512']
    num_classes = 10 
else:
    print('Unknown dataset!')

if args.version == 'RFB_vgg':
    from models.RFB_Net_vgg import build_net
elif args.version == 'RFB_E_vgg':
    from models.RFB_Net_E_vgg import build_net
elif args.version == 'RFB_mobile':
    from models.RFB_Net_mobile import build_net
    cfg = COCO_mobile_300
else:
    print('Unkown version!')

priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)
if not args.cuda:
    priors = priors.cpu()

def infer_net(save_file, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(os.path.dirname(save_file)):
        os.mkdir(os.path.dirname(save_file))
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    result = dict()

    _t = {'im_detect': Timer(), 'misc': Timer()}

    # if args.retest:
    #     f = open(det_file,'rb')
    #     all_boxes = pickle.load(f)
    #     print('Evaluating detections')
    #     testset.evaluate_detections(all_boxes, save_folder)
    #     return

    for i in range(num_images):
        # img = testset.pull_image(i)
        try:
            img = cv2.imread(testset[i])
        except:
            print('Image error')
            continue 
        x = Variable(transform(img).unsqueeze(0),volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x)      # forward pass
        boxes, scores = detector.forward(out,priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            print('=>',j)
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            print(c_dets)
            print('--------')

            if args.dataset == 'VOC':
                cpu = True
            else:
                cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            print(c_dets)
            print('--------')
            all_boxes[j][i] = c_dets
            print('=>',j,i,all_boxes[j][i])
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                print('FLAG')
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
                    # print('=>',j,i,all_boxes[j][i])

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

        print('========')
        pprint.pprint(all_boxes[1][i])
        assert 0

    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    # print('Evaluating detections')
    # testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300,512)[args.size=='512']
    # num_classes = (21, 81)[args.dataset == 'COCO']
    net = build_net('test', img_dim, num_classes)    # initialize detector
    state_dict = torch.load(args.trained_model)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    print(net)
    # load data
    if args.dataset == 'VOC':
        testset = VOCDetection(
            VOCroot, [('2007', 'test')], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        testset = COCODetection(
            COCOroot, [('2014', 'minival')], None)
            #COCOroot, [('2015', 'test-dev')], None)
    elif args.dataset == 'juggdet':
        with open(args.test_set,'r') as f: 
            testset = [os.path.join(args.test_prefix, x.strip()) for x in f.readlines()]
    else:
        print('Only VOC and COCO dataset are supported now!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # evaluation
    #top_k = (300, 200)[args.dataset == 'COCO']
    top_k = 200
    detector = Detect(num_classes,0,cfg)
    save_folder = os.path.join(args.save_folder,args.dataset)
    rgb_means = ((104, 117, 123),(103.94,116.78,123.68))[args.version == 'RFB_mobile']
    infer_net(save_folder, net, detector, args.cuda, testset,
             BaseTransform(net.size, rgb_means, (2, 0, 1)),
             top_k, thresh=args.thresh)
