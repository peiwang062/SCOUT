import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from extra_setting import *
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import cv2
import seaborn as sns
import operator
from scipy import misc

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end ade Training')
parser.add_argument('-d', '--dataset', default='ade', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='5', help='index of gpus to use')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='5', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./ade/checkpoint_pretrain_vgg16_bn.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--students', default='beginners', help='user type')
parser.add_argument('--maps', default='abs', help='explanation maps')


def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)

    # create model
    model_main = models.__dict__['vgg16_bn'](pretrained=True)
    model_main.classifier[-1] = nn.Linear(model_main.classifier[-1].in_features, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_main.module.load_state_dict(checkpoint['state_dict_m'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.students == 'beginners':
        all_correct_student = np.load('./ade/all_correct_random_te.npy')
        all_predicted_student = np.load('./ade/all_predicted_random_te.npy')
        all_gt_target_student = np.load('./ade/all_gt_target_random_te.npy')
    else:
        all_correct_student = np.load('./ade/all_correct_alexnet_te.npy')
        all_predicted_student = np.load('./ade/all_predicted_alexnet_te.npy')
        all_gt_target_student = np.load('./ade/all_gt_target_alexnet_te.npy')


    # generate predicted hardness score
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()
    prec1, prec5, all_correct_te, all_predicted_te, all_entropy_te, all_class_dis_te, all_gt_target_te = validate(
        val_loader, model_main, criterion, criterion_f)

    all_predicted_te = all_predicted_te.astype(int)
    np.save('./ade/all_correct_te_cls_vgg16.npy', all_correct_te)
    np.save('./ade/all_predicted_te_cls_vgg16.npy', all_predicted_te)
    np.save('./ade/all_entropy_te_cls_vgg16.npy', all_entropy_te)
    np.save('./ade/all_class_dis_te_cls_vgg16.npy', all_class_dis_te)
    np.save('./ade/all_gt_target_te_cls_vgg16.npy', all_gt_target_te)

    all_correct_teacher = np.load('./ade/all_correct_te_cls_vgg16.npy')
    all_predicted_teacher = np.load('./ade/all_predicted_te_cls_vgg16.npy')
    all_entropy_teacher = np.load('./ade/all_entropy_te_cls_vgg16.npy')
    all_class_dis_teacher = np.load('./ade/all_class_dis_te_cls_vgg16.npy')
    all_gt_target_teacher = np.load('./ade/all_gt_target_te_cls_vgg16.npy')

    # in order to model machine teaching, the examples we care about should be those that student network misclassified but teacher network make it
    interested_idx = np.intersect1d(np.where(all_correct_student == 0), np.where(all_correct_teacher == 1))
    predicted_class = all_predicted_student[interested_idx]
    counterfactual_class = all_gt_target_student[interested_idx]

    imlist = []
    imclass = []

    with open('./ade/ADEChallengeData2016/ADE_gt_val.txt', 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append(impath)
            imclass.append(imlabel)

    picked_list = []
    picked_class_list = []
    for i in range(np.size(interested_idx)):
        picked_list.append(imlist[interested_idx[i]])
        picked_class_list.append(imclass[interested_idx[i]])

    heat_map_hp = Heatmap_hp(model_main, target_layer_names=["42"], use_cuda=True)
    heat_map_cls = Heatmap_cls(model_main, target_layer_names=["42"], use_cuda=True)

    dis_extracted_attributes = np.load('./ade/dis_extracted_attributes.npy')

    picked_seg_list = []
    for i in range(np.size(interested_idx)):
        img_name = picked_list[i]
        img_name = img_name[:27] + "annotations" + img_name[33:-3] + "png"
        picked_seg_list.append(img_name)


    # save ade hard info
    ade_cf = './ade/ADEcf_gt_val.txt'
    fl = open(ade_cf, 'w')
    num_cf = 0
    for ii in range(len(picked_list)):
        # example_info = picked_list[ii] + " " + picked_class_list[ii] + " " + str(interested_idx[ii])
        example_info = picked_list[ii] + " " + picked_class_list[ii] + " " + str(num_cf)
        fl.write(example_info)
        fl.write("\n")
        num_cf = num_cf + 1
    fl.close()

    # data loader
    assert callable(datasets.__dict__['ade_cf'])
    get_dataset = getattr(datasets, 'ade_cf')
    num_classes = datasets._NUM_CLASSES['ade_cf']
    _, val_hard_loader = get_dataset(
        batch_size=5, num_workers=args.workers)


    print(args.students)
    print(args.maps)
    remaining_mask_size_pool = np.arange(0.1, 1.0, 0.1)
    IOU = cf_proposal_extraction(val_hard_loader, heat_map_hp, heat_map_cls,
                                                                     picked_list, dis_extracted_attributes,
                                                                     picked_seg_list, predicted_class,
                                                                     remaining_mask_size_pool, args.maps)


    print(IOU)





def validate(val_loader, model_main, criterion, criterion_f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    end = time.time()

    all_correct_te = []
    all_predicted_te = []
    all_entropy_te = []
    all_class_dis = np.zeros((1, 1040))
    all_gt_target = []
    for i, (input, target, index) in enumerate(val_loader):

        all_gt_target = np.concatenate((all_gt_target, target), axis=0)

        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)
        class_dis = F.softmax(output, dim=1)
        class_dis = class_dis.data.cpu().numpy()
        all_class_dis = np.concatenate((all_class_dis, class_dis), axis=0)

        loss = criterion(output, target)

        entropy = -1 * F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        entropy = torch.sum(entropy, dim=1).data.cpu().numpy()
        all_entropy_te = np.concatenate((all_entropy_te, entropy), axis=0)

        p_i_m = torch.max(output, dim=1)[1]
        all_predicted_te = np.concatenate((all_predicted_te, p_i_m), axis=0)
        p_i_m = p_i_m.long()
        p_i_m[p_i_m - target == 0] = -1
        p_i_m[p_i_m > -1] = 0
        p_i_m[p_i_m == -1] = 1
        correct = p_i_m.float()
        all_correct_te = np.concatenate((all_correct_te, correct), axis=0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # print(' * Testing Prec@1 {top1.avg:.3f}'.format(top1=top1))
    all_class_dis = all_class_dis[1:, :]
    return top1.avg, top5.avg, all_correct_te, all_predicted_te, all_entropy_te, all_class_dis, all_gt_target


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def largest_indices_each_example(all_response, topK):
    topK_maxIndex = np.zeros((np.size(all_response, 0), topK), dtype=np.int16)
    topK_maxValue = np.zeros((np.size(all_response, 0), topK))
    for i in range(np.size(topK_maxIndex, 0)):
        arr = all_response[i, :]
        topK_maxIndex[i, :] = np.argsort(arr)[-topK:][::-1]
        topK_maxValue[i, :] = np.sort(arr)[-topK:][::-1]
    return topK_maxIndex, topK_maxValue


def save_predicted_hardness(train_loader, val_loader, model_ahp_trunk, model_ahp_hp):
    model_ahp_trunk.eval()
    model_ahp_hp.eval()
    # hardness_scores_tr = []
    # hardness_scores_idx_tr = []
    # for i, (input, target, index) in enumerate(train_loader):
    #     input = input.cuda()
    #     predicted_hardness_scores =  model_ahp_hp(model_ahp_trunk(input)).squeeze()
    #     scores = predicted_hardness_scores.data.cpu().numpy()
    #     hardness_scores_tr = np.concatenate((hardness_scores_tr, scores), axis=0)
    #     index = index.numpy()
    #     hardness_scores_idx_tr = np.concatenate((hardness_scores_idx_tr, index), axis=0)

    hardness_scores_val = []
    hardness_scores_idx_val = []
    for i, (input, target, index) in enumerate(val_loader):
        input = input.cuda()
        trunk_output = model_ahp_trunk(input)
        predicted_hardness_scores, _ = model_ahp_hp(trunk_output)
        scores = predicted_hardness_scores.data.cpu().numpy().squeeze()
        hardness_scores_val = np.concatenate((hardness_scores_val, scores), axis=0)
        index = index.numpy()
        hardness_scores_idx_val = np.concatenate((hardness_scores_idx_val, index), axis=0)

    return hardness_scores_val, hardness_scores_idx_val


def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class FeatureExtractor_hp():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x


class FeatureExtractor_cls():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x

class ModelOutputs_hp():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_hp(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        confidence_score = F.softmax(output, dim=1)
        confidence_score = torch.max(confidence_score, dim=1)[0]
        return target_activations, confidence_score


class ModelOutputs_cls():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_cls(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        return target_activations, output



def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def show_segment_on_image(img, mask, mark_locs=None, is_cls=True):
    img = np.float32(img)
    # if is_cls == False:
    #     threshold = np.sort(mask.flatten())[-int(0.05*224*224)]
    #     mask[mask < threshold] = 0
    #     mask[mask > 0] = 1
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * mask * img)
    if is_cls == False:
        if np.sum(mark_locs) > 0:
            x, y = np.where(mark_locs == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 2, (0,0,255))
    return img



class Heatmap_hp:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs_hp(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):

        features, output = self.extractor(input)

        grads_val = torch.autograd.grad(output, features[0], grad_outputs=torch.ones_like(output),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy()

        mask_positive = np.copy(grads_val)
        mask_positive[mask_positive < 0.0] = 0.0
        mask_positive = mask_positive.squeeze()

        target = features[-1]
        target = target.cpu().data.numpy()

        cam_positive = target * mask_positive
        cam_positive = np.sum(cam_positive, axis=1)

        return cam_positive



class Heatmap_cls:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs_cls(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, CounterClass, TargetClass):

        features, output = self.extractor(input)

        target = features[-1]
        target = target.cpu().data.numpy()

        classifier_heatmaps = np.zeros((input.shape[0], np.size(target, 2), np.size(target, 2), 2))
        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(output.shape[0]), CounterClass] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy().squeeze()

        cam_positive = target * grads_val
        cam_positive = np.sum(cam_positive, axis=1)
        classifier_heatmaps[:, :, :, 0] = cam_positive

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(output.shape[0]), TargetClass] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy().squeeze()

        cam_positive = target * grads_val
        cam_positive = np.sum(cam_positive, axis=1)
        classifier_heatmaps[:, :, :, 1] = cam_positive


        return classifier_heatmaps





def cf_proposal_extraction(val_loader, heat_map_hp, heat_map_cls, imglist, dis_extracted_attributes, seg_list, predicted_class, remaining_mask_size_pool, maps):


    IOU = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))
    i_sample = 0

    for i, (input, target, index) in enumerate(val_loader):

        input = input.cuda()
        print('processing sample', i)

        easiness_heatmaps_set = heat_map_hp(input)

        classifier_heatmaps_set = heat_map_cls(input, predicted_class[index], target)

        counter_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 0]
        counterfactual_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 1]

        for i_batch in range(index.shape[0]):
            easiness_heatmaps = easiness_heatmaps_set[i_batch, :, :].squeeze()

            counter_class_heatmaps = counter_class_heatmaps_set[i_batch, :, :].squeeze()
            counterfactual_class_heatmaps = counterfactual_class_heatmaps_set[i_batch, :, :].squeeze()

            seg_img = misc.imread(seg_list[index[i_batch]])
            seg_img = np.resize(seg_img, (224, 224))

            for i_remain in range(np.size(remaining_mask_size_pool)):
                remaining_mask_size = remaining_mask_size_pool[i_remain]

                if maps == 'a':
                    cf_heatmap = counterfactual_class_heatmaps
                elif maps == 'ab':
                    cf_heatmap = (np.amax(counter_class_heatmaps) - counter_class_heatmaps) * counterfactual_class_heatmaps
                else:
                    cf_heatmap = easiness_heatmaps * (np.amax(counter_class_heatmaps) - counter_class_heatmaps) * counterfactual_class_heatmaps


                cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
                threshold = np.sort(cf_heatmap.flatten())[int(-remaining_mask_size * 224 * 224)]
                cf_heatmap[cf_heatmap > threshold] = 1
                cf_heatmap[cf_heatmap < 1] = 0

                dis_attributes = dis_extracted_attributes[target[i_batch], predicted_class[index[i_batch]]]

                if isinstance(dis_attributes, int):
                    dis_attributes = [dis_attributes]

                if len(dis_attributes) == 0:
                    IOU[i_sample, i_remain] = float('NaN')
                    continue
                dis_attributes = np.array(dis_attributes)

                commom_seg_img = np.zeros((224, 224))
                for i_com in range(np.size(dis_attributes)):
                    commom_seg_img[seg_img == dis_attributes[i_com]] = 1



                if np.sum(commom_seg_img) == 0:
                    IOU[i_sample, i_remain] = float('NaN')
                    continue
                cur_IOU = np.sum(cf_heatmap * commom_seg_img) / np.sum(cf_heatmap + commom_seg_img - cf_heatmap * commom_seg_img)
                IOU[i_sample, i_remain] = cur_IOU

            i_sample = i_sample + 1

    return np.nanmean(IOU, axis=0)



if __name__ == '__main__':
    main()



