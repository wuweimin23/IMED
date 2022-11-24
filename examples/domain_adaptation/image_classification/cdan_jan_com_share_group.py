import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan_ensemble import ImageClassifier as ImageClassifier1
from dalib.adaptation.cdan_ensemble import ConditionalDomainAdversarialLoss,Head,Combination_tra
from dalib.adaptation.jan import JointMultipleKernelMaximumMeanDiscrepancy, Theta
from dalib.adaptation.jan_ensemble import ImageClassifier as ImageClassifier2
from dalib.modules.kernels import GaussianKernel
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from GradCAM import GradCAM

sys.path.append('.')
import utils_ensemble as utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True


    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source, args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader,device = device)
    train_target_iter = ForeverDataIterator(train_target_loader,device = device)

    # create model
    print("=> using model '{}'".format(args.arch))
    backbone_cdan = utils.get_model(args.arch, pretrain=not args.scratch)    
    backbone_jan = utils.get_model(args.arch, pretrain=not args.scratch)  
    pool_layer_cdan = nn.Identity() if args.no_pool else None
    classifier_cdan = ImageClassifier1(backbone_cdan, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer_cdan, finetune=not args.scratch).to(device)   
    classifier_jan = ImageClassifier2(backbone_cdan, num_classes, bottleneck_dim=args.bottleneck_dim,
                                  pool_layer=pool_layer_cdan, finetune=not args.scratch).to(device) 
    head = Head(args.bottleneck_dim, num_classes).to(device)
    head_true = Head(args.bottleneck_dim, num_classes).to(device)
    


    # define loss function
    classifier_feature_dim = classifier_cdan.features_dim
    W = utils.W_generate(classifier_feature_dim, args.group_num).to(device)
    combination = Combination_tra(args.bottleneck_dim, num_classes, W, args.group_num).to(device)
    mcc_loss = MinimumClassConfusionLoss(temperature=args.temperature)

    if args.randomized:
        domain_discri = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        domain_discri2 = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
        domain_discri2 = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    all_parameters_cdan = classifier_cdan.get_parameters() + domain_discri.get_parameters()
    # define optimizer and lr scheduler
    optimizer_cdan = SGD(all_parameters_cdan, args.lr, momentum=args.momentum, weight_decay=args.wd_cdan, nesterov=True)
    lr_scheduler_cdan = LambdaLR(optimizer_cdan, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    #Head
    parameters_head = head.get_parameters()
    optimizer_head = SGD(parameters_head,args.lr, momentum=args.momentum, weight_decay=args.wd_cdan, nesterov=True)
    lr_scheduler_head = LambdaLR(optimizer_head, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    base_optimizer = torch.optim.SGD
    parameters_com = head_true.get_parameters() + combination.get_parameters()
    ad_optimizer3 = SAM(parameters_com, base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_discri = SGD(domain_discri2.get_parameters(),args.lr, momentum=args.momentum, weight_decay=args.wd_cdan, nesterov=True)
    lr_scheduler_ad3 = LambdaLR(ad_optimizer3, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_discri = LambdaLR(optimizer_discri, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


    # define loss function
    domain_adv_cdan = ConditionalDomainAdversarialLoss(
        domain_discri, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    domain_adv_cdan2 = ConditionalDomainAdversarialLoss(
        domain_discri2, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    ###jan loss

    if args.adversarial:
        thetas = [Theta(dim).to(device) for dim in (classifier_cdan.features_dim, num_classes)]
    else:
        thetas = None   
    jmmd_loss_jan = JointMultipleKernelMaximumMeanDiscrepancy(
        kernels=(
            [GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
            (GaussianKernel(sigma=0.92, track_running_stats=False),)
        ),
        linear=args.linear, thetas=thetas
    ).to(device)

    parameters = classifier_jan.get_parameters()
    if thetas is not None:
        parameters += [{"params": theta.parameters(), 'lr': 0.1} for theta in thetas]

    #define optimizer
    optimizer_jan = SGD(parameters, args.lr, momentum=args.momentum, weight_decay=args.wd_jan, nesterov=True)
    lr_scheduler_jan = LambdaLR(optimizer_jan, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))   

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier_cdan,classifier_jan, head, head_true,combination, domain_adv_cdan, domain_adv_cdan2, optimizer_cdan,
              lr_scheduler_cdan, jmmd_loss_jan,optimizer_jan,lr_scheduler_jan, optimizer_head, lr_scheduler_head,ad_optimizer3, optimizer_discri, lr_scheduler_ad3, lr_scheduler_discri , mcc_loss,W, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier_cdan,classifier_jan, head,head_true,combination,num_classes,classifier_feature_dim, W,args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier_cdan.state_dict(), logger.get_checkpoint_path('cdan_latest'))
        torch.save(classifier_jan.state_dict(), logger.get_checkpoint_path('jan_latest'))
        torch.save(head.state_dict(), logger.get_checkpoint_path('head_latest'))
        torch.save(head_true.state_dict(), logger.get_checkpoint_path('head_true_latest'))
        torch.save(combination.state_dict(), logger.get_checkpoint_path('combination_latest'))
  

        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('cdan_latest'), logger.get_checkpoint_path('cdan_best'))
            shutil.copy(logger.get_checkpoint_path('jan_latest'), logger.get_checkpoint_path('jan_best'))
            shutil.copy(logger.get_checkpoint_path('head_latest'), logger.get_checkpoint_path('head_best'))
            shutil.copy(logger.get_checkpoint_path('head_true_latest'), logger.get_checkpoint_path('head_true_best'))
            shutil.copy(logger.get_checkpoint_path('combination_latest'), logger.get_checkpoint_path('combination_best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # train student model
    classifier_cdan.load_state_dict(torch.load(logger.get_checkpoint_path('cdan_best')))
    classifier_jan.load_state_dict(torch.load(logger.get_checkpoint_path('jan_best')))
    head.load_state_dict(torch.load(logger.get_checkpoint_path('head_best')))
    head_true.load_state_dict(torch.load(logger.get_checkpoint_path('head_true_best')))
    combination.load_state_dict(torch.load(logger.get_checkpoint_path('combination_best')))


    student_model = ImageClassifier1(backbone_cdan, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer_cdan, finetune=not args.scratch).to(device)
    student_head = Head(args.bottleneck_dim, num_classes).to(device)

    parameters_student = student_model.get_parameters() + student_head.get_parameters()
    optimizer_student = SGD(parameters_student,args.lr, momentum=args.momentum, weight_decay=args.wd_cdan, nesterov=True)
    lr_scheduler_student = LambdaLR(optimizer_student, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


    print('start training student model ************************************')
    best_acc1 = 0
    for epoch in range(args.distill_epochs):
        train_student(train_source_iter, train_target_iter, classifier_cdan,classifier_jan, head, head_true,combination, student_model, student_head, optimizer_student, lr_scheduler_student, W, epoch, args)

        # evaluate on validation set
        acc1 = utils.validate_student(val_loader, student_model, student_head,num_classes,classifier_feature_dim,args, device)

        torch.save(student_model.state_dict(), logger.get_checkpoint_path('student_model_latest'))
        torch.save(student_head.state_dict(), logger.get_checkpoint_path('student_head_latest'))
  
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('student_model_latest'), logger.get_checkpoint_path('student_model_best'))
            shutil.copy(logger.get_checkpoint_path('student_head_latest'), logger.get_checkpoint_path('student_head_best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))


    student_model.load_state_dict(torch.load(logger.get_checkpoint_path('student_model_best')))
    student_head.load_state_dict(torch.load(logger.get_checkpoint_path('student_head_best')))

    student_model.train()
    student_head.train()
    vis = GradCAM(student_model, student_head, device)
    count = 0
    for i, (x_t, labels_t) in enumerate(test_loader):
        x_t = x_t.to(device)
        labels_t = labels_t.to(device)
        vis(x_t, labels_t, count)
        count += x_t.shape[0]
              
    acc1 = utils.validate_student(test_loader, student_model, student_head,num_classes,classifier_feature_dim, args, device)   #需传入cdan的鉴别器domain_discri


    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train_student(train_source_iter, train_target_iter, classifier_cdan,classifier_jan, head, head_true,combination, student_model, student_head, optimizer_student, lr_scheduler_student, W, epoch, args):
    classifier_cdan.eval()
    classifier_jan.eval()
    head.eval()
    head_true.eval()
    combination.eval()
    student_model.train()
    student_head.train()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        x = torch.cat((x_s, x_t), dim=0)

        #combination
        with torch.no_grad():
            f_cdan = classifier_cdan(x)
            f_jan = classifier_jan(x)
            y_cdan = head(f_cdan)
            y_jan = head(f_jan)
            f_com = combination(f_cdan,y_cdan,f_jan,y_jan)
            y_com = head_true(f_com)
        
        f_student = student_model(x)
        y_student = student_head(f_student)

        y_student_s,_ = y_student.chunk(2, dim=0) 
        acc_loss = F.cross_entropy(y_student_s, labels_s)
        tea_stu_loss = utils.CrossEntropy(y_student,y_com)
        tea_stu_loss_feature = utils.CrossEntropy(f_student,f_com)
        loss = acc_loss + tea_stu_loss + tea_stu_loss_feature

        optimizer_student.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_student.step()
        lr_scheduler_student.step()

        if i % args.print_freq == 0:
            print('Epoch:{},acc_loss:{},tea_stu_loss\n'.format(i,tea_stu_loss))



def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator, model1: ImageClassifier1, model2: ImageClassifier1,
          model_head:Head,model_head_true:Head, model_combination ,domain_adv: ConditionalDomainAdversarialLoss, domain_adv2: ConditionalDomainAdversarialLoss, optimizer_cdan: SGD,
          lr_scheduler_cdan: LambdaLR, jmmd_loss: JointMultipleKernelMaximumMeanDiscrepancy, optimizer_jan: SGD,
          lr_scheduler_jan: LambdaLR, optimizer_head: SGD, lr_scheduler_head: LambdaLR, ad_optimizer3, optimizer_discri, lr_scheduler_ad3, lr_scheduler_discri , mcc_loss,W,epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_cdan = AverageMeter('Loss cdan', ':3.2f')
    trans_losses_cdan = AverageMeter('Trans Loss cdan', ':5.4f')
    cls_accs_cdan = AverageMeter('Cls Acc cdan', ':3.1f')
    domain_accs_cdan = AverageMeter('Domain Acc cdan', ':3.1f')
    losses_jan = AverageMeter('Loss jan', ':3.2f')
    trans_losses_jan = AverageMeter('Trans Loss jan', ':5.4f')
    cls_accs_jan = AverageMeter('Cls Acc jan', ':3.1f')
    tgt_accs_jan = AverageMeter('Tgt Acc jan', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_cdan, trans_losses_cdan, cls_accs_cdan, domain_accs_cdan, losses_jan, trans_losses_jan, cls_accs_jan, tgt_accs_jan],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model1.train()
    model2.train()
    model_head.train()
    model_head_true.train()
    model_combination.train()
    domain_adv.train()
    domain_adv2.train()
    jmmd_loss.train()


    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, labels_t = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)

        # measure data loading time
        data_time.update(time.time() - end)


        #####################################
        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        f_cdan = model1(x)  
        y = model_head(f_cdan)
        y_s, y_t = y.chunk(2, dim=0)   
        f_s, f_t = f_cdan.chunk(2, dim=0)

        #cdan
        cls_loss_cdan = F.cross_entropy(y_s, labels_s)  
        transfer_loss_cdan = domain_adv(y_s, f_s, y_t, f_t)   
        domain_acc_cdan = domain_adv.domain_discriminator_accuracy  
        loss_cdan = cls_loss_cdan + transfer_loss_cdan * args.trade_off

        cls_acc_cdan = accuracy(y_s, labels_s)[0]  

        losses_cdan.update(loss_cdan.item(), x_s.size(0))
        cls_accs_cdan.update(cls_acc_cdan, x_s.size(0))
        domain_accs_cdan.update(domain_acc_cdan, x_s.size(0))
        trans_losses_cdan.update(transfer_loss_cdan.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer_cdan.zero_grad()
        optimizer_head.zero_grad()
        loss_cdan.backward()
        optimizer_cdan.step()
        optimizer_head.step()
        lr_scheduler_cdan.step()
        lr_scheduler_head.step()

        ###jan
        # compute output
        f_cdan = model1(x)  
        f = model2(x)  
        y = model_head(f)
        y_s, y_t = y.chunk(2, dim=0)   
        f_s, f_t = f.chunk(2, dim=0)


        cls_loss_jan = F.cross_entropy(y_s, labels_s)
        transfer_loss_jan = jmmd_loss(
            (f_s, F.softmax(y_s, dim=1)),
            (f_t, F.softmax(y_t, dim=1))
        ) 

        loss_jan = cls_loss_jan + transfer_loss_jan * args.trade_off 

        cls_acc_jan = accuracy(y_s, labels_s)[0]
        tgt_acc_jan = accuracy(y_t, labels_t)[0]

        losses_jan.update(loss_jan.item(), x_s.size(0))
        cls_accs_jan.update(cls_acc_jan.item(), x_s.size(0))
        tgt_accs_jan.update(tgt_acc_jan.item(), x_t.size(0))
        trans_losses_jan.update(transfer_loss_jan.item(), x_s.size(0))

        # compute gradient and do SGD step
        optimizer_jan.zero_grad()
        optimizer_head.zero_grad()
        loss_jan.backward()
        optimizer_jan.step()
        optimizer_head.step()
        lr_scheduler_jan.step()
        lr_scheduler_head.step()

        ad_optimizer3.zero_grad()
        optimizer_discri.zero_grad()

        with torch.no_grad():
            f_cdan = model1(x)
            y_cdan = model_head(f_cdan)
            f_jan = model2(x)
            y_jan = model_head(f_jan)
        
        f_com = model_combination(f_cdan,y_cdan,f_jan,y_jan)
        y_com = model_head_true(f_com)
        y_s, y_t = y_com.chunk(2, dim=0)  
        f_s, f_t = f_com.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        mcc_loss_value = mcc_loss(y_t)
        loss = cls_loss + mcc_loss_value
        loss.backward()

        ad_optimizer3.first_step(zero_grad=True)

        with torch.no_grad():
            f_cdan = model1(x)
            y_cdan = model_head(f_cdan)
            f_jan = model2(x)
            y_jan = model_head(f_jan)
        
        f_com = model_combination(f_cdan,y_cdan,f_jan,y_jan)
        y_com = model_head_true(f_com)
        y_s, y_t = y_com.chunk(2, dim=0)   
        f_s, f_t = f_com.chunk(2, dim=0)

        loss_class = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv2(y_s, f_s, y_t, f_t) + mcc_loss(y_t)  
        loss = loss_class + transfer_loss
        
        loss.backward()
        ad_optimizer3.second_step(zero_grad=True)
        optimizer_discri.step()
        lr_scheduler_ad3.step()
        lr_scheduler_discri.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='JAN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', default = 'W', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', default = 'A', help='target domain(s)', nargs='+')
    parser.add_argument('--train-resizing', type=str, default='default')
    parser.add_argument('--val-resizing', type=str, default='default')
    parser.add_argument('--resize-size', type=int, default=224,
                        help='the image size after resizing')
    parser.add_argument('--no-hflip', action='store_true',
                        help='no random horizontal flipping during training')
    parser.add_argument('--norm-mean', type=float, nargs='+',
                        default=(0.485, 0.456, 0.406), help='normalization mean')
    parser.add_argument('--norm-std', type=float, nargs='+',
                        default=(0.229, 0.224, 0.225), help='normalization std')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False, action='store_true', help='use entropy conditioning')
    parser.add_argument('--linear', default=False, action='store_true',
                        help='whether use the linear version')
    parser.add_argument('--adversarial', default=False, action='store_true',
                        help='whether use adversarial theta')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate',  default=0.003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.0003, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd_jan', '--weight-decay-jan', default=0.0005, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--wd_cdan', '--weight-decay-cdan', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
                        # dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--distill_epochs', default=5, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='logs/jan/Office31_W2A',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument("--group_num", type=int, default='8', 
                        help="group linear num")
    parser.add_argument("--threshold", type=float, default='0.06', 
                        help="threshold for pseudo label")
    parser.add_argument("--threshold_exist", type=int, default='1', 
                        help="threshold existence")
    parser.add_argument('--temperature', default=2.5, type=float, help='parameter temperature scaling')
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument("--model_num", type=int, default='1', 
                        help="1 is new_1, 2 is new_2, 0 is tra")
    args = parser.parse_args()
    main(args)


