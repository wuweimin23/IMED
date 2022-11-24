# Credits: https://github.com/thuml/Transfer-Learning-Library
import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp
import os
import wandb

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity   #calculate the cosine similarity

sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.cdan_ensemble import ConditionalDomainAdversarialLoss, ImageClassifier, Combination_Korn, Head
from dalib.adaptation.cdan_ensemble import ImageClassifier as ImageClassifier_1
from dalib.adaptation.mcc import MinimumClassConfusionLoss
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from common.utils.sam import SAM

sys.path.append('.')
import utils_ensemble as utils


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
    device = args.device

    # Data loading code
    train_transform = utils.get_train_transform(args.train_resizing, random_horizontal_flip=not args.no_hflip,
                                                random_color_jitter=False, resize_size=args.resize_size,
                                                norm_mean=args.norm_mean, norm_std=args.norm_std)
    val_transform = utils.get_val_transform(args.val_resizing, resize_size=args.resize_size,
                                            norm_mean=args.norm_mean, norm_std=args.norm_std)
    print("train_transform: ", train_transform)
    print("val_transform: ", val_transform)

    train_source_dataset, train_target_dataset, val_dataset, test_dataset, num_classes, args.class_names = \
        utils.get_dataset(args.data, args.root, args.source,
                          args.target, train_transform, val_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using model '{}'".format(args.arch))
    #模型一
    backbone_cdan1 = utils.get_model(args.arch, pretrain=not args.scratch)
    pool_layer1 = nn.Identity() if args.no_pool else None
    classifier1 = ImageClassifier_1(backbone_cdan1, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer1, finetune=not args.scratch).to(device)
    classifier_feature_dim = classifier1.features_dim

    head_true = Head(args.bottleneck_dim, num_classes).to(device)


    random.seed(args.seed2)
    torch.manual_seed(args.seed2)
    cudnn.deterministic = True
    backbone_cdan2 = utils.get_model(args.arch, pretrain=not args.scratch)    #jan
    pool_layer2 = nn.Identity() if args.no_pool else None
    classifier2 = ImageClassifier_1(backbone_cdan2, num_classes, bottleneck_dim=args.bottleneck_dim,
                                  pool_layer=pool_layer2, finetune=not args.scratch).to(device) 


    combination = Combination_Korn(args.bottleneck_dim, num_classes).to(device)

    if args.randomized:
        domain_discri1 = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        domain_discri2 = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
        domain_discri_com = DomainDiscriminator(args.randomized_dim, hidden_size=1024).to(device)
    else:
        domain_discri1 = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
        domain_discri2 = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)
        domain_discri_com = DomainDiscriminator(classifier_feature_dim * num_classes, hidden_size=1024).to(device)

    # define optimizer and lr scheduler
    base_optimizer = torch.optim.SGD
    ad_optimizer1 = SGD(domain_discri1.get_parameters(), 
        args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer1_class = SAM(classifier1.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    ad_optimizer2 = SGD(domain_discri2.get_parameters(), 
        args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer2_class = SAM(classifier2.get_parameters(), base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    parameters_com = head_true.get_parameters() + combination.get_parameters()
    ad_optimizer3 = SAM(parameters_com, base_optimizer, rho=args.rho, adaptive=False,
                    lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    optimizer_discri = SGD(domain_discri_com.get_parameters(),args.lr, momentum=args.momentum, weight_decay=args.wd_cdan, nesterov=True)

    
    lr_scheduler1_class = LambdaLR(optimizer1_class, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad1 = LambdaLR(ad_optimizer1, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler2_class = LambdaLR(optimizer2_class, lambda x: args.lr *
                            (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad2 = LambdaLR(ad_optimizer2, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_ad3 = LambdaLR(ad_optimizer3, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    lr_scheduler_discri = LambdaLR(optimizer_discri, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    

    # define loss function
    domain_adv1 = ConditionalDomainAdversarialLoss(
        domain_discri1, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    domain_adv2 = ConditionalDomainAdversarialLoss(
        domain_discri2, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    domain_adv_com = ConditionalDomainAdversarialLoss(
        domain_discri_com, entropy_conditioning=args.entropy,
        num_classes=num_classes, features_dim=classifier_feature_dim, randomized=args.randomized,
        randomized_dim=args.randomized_dim
    ).to(device)

    mcc_loss1 = MinimumClassConfusionLoss(temperature=args.temperature)
    mcc_loss2 = MinimumClassConfusionLoss(temperature=args.temperature)
    mcc_loss3 = MinimumClassConfusionLoss(temperature=args.temperature)


    best_acc1 = 0
    for epoch in range(args.epochs):
        # train for one epoch

        train(train_source_iter, train_target_iter, classifier1,classifier2, domain_adv1,domain_adv2, mcc_loss1,mcc_loss2, mcc_loss3, combination,head_true,domain_adv_com,
            optimizer1_class, ad_optimizer1, optimizer2_class, ad_optimizer2, ad_optimizer3, optimizer_discri,
            lr_scheduler1_class, lr_scheduler_ad1, lr_scheduler2_class, lr_scheduler_ad2, lr_scheduler_ad3, lr_scheduler_discri, epoch, args)
        
        # evaluate on validation set
        acc1 = utils.validate(val_loader, classifier1, classifier2, head_true, combination,num_classes,classifier_feature_dim, args, device)

        # remember best acc@1 and save checkpoint
        torch.save(classifier1.state_dict(), logger.get_checkpoint_path('cdan_latest'))
        torch.save(classifier2.state_dict(), logger.get_checkpoint_path('jan_latest'))
        torch.save(head_true.state_dict(), logger.get_checkpoint_path('head_true_latest'))
        torch.save(combination.state_dict(), logger.get_checkpoint_path('combination_latest'))
  

        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('cdan_latest'), logger.get_checkpoint_path('cdan_best'))
            shutil.copy(logger.get_checkpoint_path('jan_latest'), logger.get_checkpoint_path('jan_best'))
            shutil.copy(logger.get_checkpoint_path('head_true_latest'), logger.get_checkpoint_path('head_true_best'))
            shutil.copy(logger.get_checkpoint_path('combination_latest'), logger.get_checkpoint_path('combination_best'))
        best_acc1 = max(acc1, best_acc1)


    print("best_acc1 = {:3.1f}".format(best_acc1))

    # train student model
    classifier1.load_state_dict(torch.load(logger.get_checkpoint_path('cdan_best')))
    classifier2.load_state_dict(torch.load(logger.get_checkpoint_path('jan_best')))
    head_true.load_state_dict(torch.load(logger.get_checkpoint_path('head_true_best')))
    combination.load_state_dict(torch.load(logger.get_checkpoint_path('combination_best')))

    backbone = utils.get_model(args.arch, pretrain=not args.scratch)    #jan
    pool_layer = nn.Identity() if args.no_pool else None
    student_model = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim,
                                 pool_layer=pool_layer, finetune=not args.scratch).to(device)
    student_head = Head(args.bottleneck_dim, num_classes).to(device)

    parameters_student = student_model.get_parameters() + student_head.get_parameters()
    optimizer_student = SGD(parameters_student,args.lr, momentum=args.momentum, weight_decay=args.wd_cdan, nesterov=True)
    lr_scheduler_student = LambdaLR(optimizer_student, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))


    print('start training student model ************************************')
    best_acc1 = 0
    for epoch in range(6):
        train_student(train_source_iter, train_target_iter, classifier1,classifier2, head_true,combination, student_model, student_head, optimizer_student, lr_scheduler_student, epoch, args)

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
              
    acc1 = utils.validate_student(test_loader, student_model, student_head,num_classes,classifier_feature_dim, args, device)   #需传入cdan的鉴别器domain_discri

    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()

def train_student(train_source_iter, train_target_iter, classifier_cdan,classifier_jan,  head_true,combination, student_model, student_head, optimizer_student, lr_scheduler_student, epoch, args):
    classifier_cdan.eval()
    classifier_jan.eval()
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
            y_cdan, f_cdan = classifier_cdan(x)
            y_jan, f_jan = classifier_jan(x)
            f_com = combination(f_cdan,y_cdan,f_jan,y_jan)
            y_com = head_true(f_com)
        
        f_student = student_model(x)
        y_student = student_head(f_student)

        tem = args.tem
        reg1 = 0.5
        reg2 = 0.5
        y_student_s,_ = y_student.chunk(2, dim=0) 
        acc_loss = F.cross_entropy(y_student_s, labels_s)
        tea_stu_loss = utils.CrossEntropy(y_student/tem,y_com/tem)
        tea_stu_loss_feature = utils.CrossEntropy(f_student/tem,f_com/tem)

        loss = acc_loss + reg1 * tea_stu_loss*tem*tem + reg2 * tea_stu_loss_feature*tem*tem

        optimizer_student.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_student.step()
        lr_scheduler_student.step()

        if i % args.print_freq == 0:
            print('Epoch:{},acc_loss:{},tea_stu_loss\n'.format(i,tea_stu_loss))

def train(train_source_iter, train_target_iter, classifier1,classifier2, domain_adv1,domain_adv2, mcc_loss1,mcc_loss2,mcc_loss3, combination,head_true,domain_adv_com,
            optimizer1_class, ad_optimizer1, optimizer2_class, ad_optimizer2, ad_optimizer3, optimizer_discri,
            lr_scheduler1_class, lr_scheduler_ad1, lr_scheduler2_class, lr_scheduler_ad2, lr_scheduler_ad3, lr_scheduler_discri, epoch, args):
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
    classifier1.train()
    classifier2.train()
    head_true.train()
    combination.train()
    domain_adv1.train()
    domain_adv2.train()
    domain_adv_com.train()

    end = time.time()

    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        x_t, _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)
        optimizer1_class.zero_grad()
        ad_optimizer1.zero_grad()
        

        # # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = classifier1(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        mcc_loss_value = mcc_loss1(y_t)
        loss = cls_loss + mcc_loss_value

        loss.backward()

        # Calculate ϵ̂ (w) and add it to the weights
        optimizer1_class.first_step(zero_grad=True)

        # Calculate task loss and domain loss
        y, f = classifier1(x)
        if args.diversity:
            y_1 = y.detach()
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv1(y_s, f_s, y_t, f_t) + mcc_loss1(y_t)
        domain_acc = domain_adv1.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses_cdan.update(loss.item(), x_s.size(0))
        cls_accs_cdan.update(cls_acc.item(), x_s.size(0))
        domain_accs_cdan.update(domain_acc.item(), x_s.size(0))
        trans_losses_cdan.update(transfer_loss.item(), x_s.size(0))

        loss.backward()
        # Update parameters of domain classifier
        ad_optimizer1.step()
        # Update parameters (Sharpness-Aware update)
        optimizer1_class.second_step(zero_grad=True)
        lr_scheduler1_class.step()
        lr_scheduler_ad1.step()

        ###classifier2
        optimizer2_class.zero_grad()
        ad_optimizer2.zero_grad()

        # compute output
        y, f = classifier2(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)
        cls_loss = F.cross_entropy(y_s, labels_s)
        mcc_loss_value = mcc_loss2(y_t)
        loss = cls_loss + mcc_loss_value 

        loss.backward()

        # Calculate ϵ̂ (w) and add it to the weights
        optimizer2_class.first_step(zero_grad=True)

        # Calculate task loss and domain loss
        y, f = classifier2(x)
        y_s, y_t = y.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = domain_adv2(y_s, f_s, y_t, f_t) + mcc_loss2(y_t)
        domain_acc = domain_adv2.domain_discriminator_accuracy
        loss = cls_loss + transfer_loss * args.trade_off

        cls_acc = accuracy(y_s, labels_s)[0]

        losses_jan.update(loss.item(), x_s.size(0))
        cls_accs_jan.update(cls_acc.item(), x_s.size(0))
        tgt_accs_jan.update(domain_acc.item(), x_s.size(0))
        trans_losses_jan.update(transfer_loss.item(), x_s.size(0))

        loss.backward()
        # Update parameters of domain classifier
        ad_optimizer2.step()
        # Update parameters (Sharpness-Aware update)
        optimizer2_class.second_step(zero_grad=True)
        lr_scheduler2_class.step()
        lr_scheduler_ad2.step()

        #combination
        if epoch >-1: 
            ad_optimizer3.zero_grad()
            optimizer_discri.zero_grad()

            with torch.no_grad():
                y_cdan, f_cdan = classifier1(x)
                y_jan, f_jan = classifier2(x)
            
            f_com = combination(f_cdan,y_cdan,f_jan,y_jan)
            y_com = head_true(f_com)
            y_s, y_t = y_com.chunk(2, dim=0)   
            f_s, f_t = f_com.chunk(2, dim=0)

            cls_loss = F.cross_entropy(y_s, labels_s)
            mcc_loss_value = mcc_loss3(y_t)
            loss = cls_loss + mcc_loss_value*args.trade_off1
            loss.backward()

            ad_optimizer3.first_step(zero_grad=True)

            with torch.no_grad():
                y_cdan, f_cdan = classifier1(x)
                y_jan, f_jan = classifier2(x)
            
            f_com = combination(f_cdan,y_cdan,f_jan,y_jan)
            y_com = head_true(f_com)
            y_s, y_t = y_com.chunk(2, dim=0)  
            f_s, f_t = f_com.chunk(2, dim=0)

            loss_class = F.cross_entropy(y_s, labels_s)
            transfer_loss = domain_adv_com(y_s, f_s, y_t, f_t) + mcc_loss3(y_t)  
            loss = loss_class + transfer_loss * args.trade_off2
            
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
    parser = argparse.ArgumentParser(
        description='CDAN+MCC with SDAT for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31', choices=utils.get_dataset_names(),
                        help='dataset: ' + ' | '.join(utils.get_dataset_names()) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)', nargs='+')
    parser.add_argument('-t', '--target', help='target domain(s)', nargs='+')
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
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=utils.get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(utils.get_model_names()) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--no-pool', default = False, action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true',
                        help='whether train from scratch.')
    parser.add_argument('-r', '--randomized', action='store_true',
                        help='using randomized multi-linear-map (default: False)')
    parser.add_argument('-rd', '--randomized-dim', default=1024, type=int,
                        help='randomized dimension when using randomized multi-linear-map (default: 1024)')
    parser.add_argument('--entropy', default=False,
                        action='store_true', help='use entropy conditioning')
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off1', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--trade-off2', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    parser.add_argument('--tem', default=3., type=float,
                        help='temperature')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75,
                        type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9,
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--seed2', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument("--diversity", default=True, 
                        help="whether use diversity loss")
    parser.add_argument('--log_results', default = False, action='store_true',
                        help="To log results in wandb")
    parser.add_argument('--gpu', type=str, default="0", help="GPU ID")
    parser.add_argument('--log_name', type=str,
                        default="log", help="log name for wandb")
    parser.add_argument('--rho', type=float, default=0.05, help="GPU ID")
    parser.add_argument('--temperature', default=2.0,
                        type=float, help='parameter temperature scaling')
    parser.add_argument('--wd_cdan', '--weight-decay-cdan', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    main(args)


