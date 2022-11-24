"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from turtle import forward
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

from common.modules.classifier_dynical import Classifier as ClassifierBase
from common.utils.metric import binary_accuracy
from ..modules.grl import WarmStartGradientReverseLayer
from ..modules.entropy import entropy


__all__ = ['ConditionalDomainAdversarialLoss', 'ImageClassifier']


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(T(f_j^t, g_j^t))],\\

    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.

    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.

    Examples::

        >>> from dalib.modules.domain_discriminator import DomainDiscriminator
        >>> from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: Optional[bool] = False,
                 randomized: Optional[bool] = False, num_classes: Optional[int] = -1,
                 features_dim: Optional[int] = -1, randomized_dim: Optional[int] = 1024,
                 reduction: Optional[str] = 'mean'):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.entropy_conditioning = entropy_conditioning

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()

        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        f = torch.cat((f_s, f_t), dim=0)
        g = torch.cat((g_s, g_t), dim=0)
        g = F.softmax(g, dim=1).detach()
        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)
        g_s_1 = torch.ones((g_s.size(0), 1))
        g_t_0 = torch.zeros((g_t.size(0), 1))
        d_label = torch.cat((g_s_1,g_t_0),dim=0)
        # print('d_label',d_label)
       
      
        weight = 1.0 + torch.exp(-entropy(g))
        batch_size = f.size(0)
        weight = weight / torch.sum(weight) * batch_size
        d_label = d_label.to(device = f_s.device)
        self.domain_discriminator_accuracy = binary_accuracy(d, d_label)
        bce = self.bce(d, d_label, weight.view_as(d))
        return bce


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)    

class Head(nn.Module):
    def __init__(self,features_dim:int, num_classes:int):   #features+dim 即为bottleneck_dim
        super(Head,self).__init__()
        self._features_dim = features_dim
        self.num_classes = num_classes
        width =1024
        self.head = nn.Sequential(
            nn.Linear(features_dim,width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width,num_classes)
        )
    
    def forward(self,x:torch.Tensor):
        predictions = self.head(x)
        return predictions

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#group_num = 128, there are two layers in fusion sub-network
class Combination_Korn(nn.Module):
    def __init__(self,features_dim:int,num_classes:int):
        super(Combination_Korn,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes

        hidden_size = 1024

        if (features_dim*num_classes) > 4096:
            assert num_classes > 0 and features_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, 1024)
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*4),
                nn.Sigmoid()
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(2*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*2),
                nn.Sigmoid()
            )
        else:
            self.map = MultiLinearMap()
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*4),
                nn.Sigmoid()
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,self.feature_dim*2),
                nn.Sigmoid()
            )

        self.relu3 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor):
        label1 = F.softmax(label1, dim=1)
        h1 = self.map(feature1, label1)

        label2 = F.softmax(label2, dim=1)
        h2 = self.map(feature2, label2)

        input = torch.cat((h1,h2),dim=1)

        weight1 = self.linear_weight1(input)
    
        weight2 = self.linear_weight2(input)

        #linear1
        feature_1_1 = feature1 * weight1[:,:self.feature_dim]
        feature_1_2 = feature1 * weight1[:,self.feature_dim:2*self.feature_dim]
        feature_1_3 = feature2 * weight1[:,self.feature_dim*2:self.feature_dim*3]
        feature_1_4 = feature2 * weight1[:,self.feature_dim*3:self.feature_dim*4]

        feature_1 = feature_1_1+feature_1_3
        feature_2 = feature_1_2+feature_1_4
        
        feature_1 = self.relu3(feature_1)
        feature_2 = self.relu3(feature_2)

        feature_2_1 = feature_1 * weight2[:,:self.feature_dim]
        feature_2_2 = feature_2 * weight2[:,self.feature_dim:2*self.feature_dim]

        feature  = feature_2_1 + feature_2_2

        return feature 
    
    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#group_num = 128, there is one layer in fusion sub-network
class Combination_Korn_1(nn.Module):
    def __init__(self,features_dim:int,num_classes:int):
        super(Combination_Korn_1,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes

        hidden_size = 1024

        if (features_dim*num_classes) > 4096:
            assert num_classes > 0 and features_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, 1024)
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*2),
                nn.Sigmoid()
            )
        else:
            self.map = MultiLinearMap()
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*2),
                nn.Sigmoid()
            )
        self.relu3 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor):
        label1 = F.softmax(label1, dim=1)
        h1 = self.map(feature1, label1)

        label2 = F.softmax(label2, dim=1)
        h2 = self.map(feature2, label2)

        input = torch.cat((h1,h2),dim=1)

        weight1 = self.linear_weight1(input)

        #linear1
        feature_1_1 = feature1 * weight1[:,:self.feature_dim]
        feature_1_2 = feature1 * weight1[:,self.feature_dim:2*self.feature_dim]

        feature  = feature_1_1 + feature_1_2

        return feature 
    
    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params


#group_num = 128, there are three layers in fusion sub-network
class Combination_Korn_3(nn.Module):
    def __init__(self,features_dim:int,num_classes:int):
        super(Combination_Korn_3,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes

        hidden_size = 1024

        if (features_dim*num_classes) > 4096:
            assert num_classes > 0 and features_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, 1024)
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*4),
                nn.Sigmoid()
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(2*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*4),
                nn.Sigmoid()
            )
            self.linear_weight3 = nn.Sequential(
                nn.Linear(2*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*2),
                nn.Sigmoid()
            )
        else:
            self.map = MultiLinearMap()
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*4),
                nn.Sigmoid()
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*4),
                nn.Sigmoid()
            )
            self.linear_weight3 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,self.feature_dim*2),
                nn.Sigmoid()
            )

        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor):
        label1 = F.softmax(label1, dim=1)
        h1 = self.map(feature1, label1)

        label2 = F.softmax(label2, dim=1)
        h2 = self.map(feature2, label2)

        input = torch.cat((h1,h2),dim=1)

        weight1 = self.linear_weight1(input)
    
        weight2 = self.linear_weight2(input)

        weight3 = self.linear_weight3(input)

        #linear1
        feature_1_1 = feature1 * weight1[:,:self.feature_dim]
        feature_1_2 = feature1 * weight1[:,self.feature_dim:2*self.feature_dim]
        feature_1_3 = feature2 * weight1[:,self.feature_dim*2:self.feature_dim*3]
        feature_1_4 = feature2 * weight1[:,self.feature_dim*3:self.feature_dim*4]

        feature_1 = feature_1_1+feature_1_3
        feature_2 = feature_1_2+feature_1_4
        
        feature_1 = self.relu3(feature_1)
        feature_2 = self.relu3(feature_2)

        feature_2_1 = feature_1 * weight2[:,:self.feature_dim]
        feature_2_2 = feature_1 * weight2[:,self.feature_dim:2*self.feature_dim]
        feature_2_3 = feature_2 * weight2[:,self.feature_dim*2:self.feature_dim*3]
        feature_2_4 = feature_2 * weight2[:,self.feature_dim*3:self.feature_dim*4]

        feature_1  = feature_2_1 + feature_2_3
        feature_2  = feature_2_2 + feature_2_4

        feature_3_1 = feature_1 * weight3[:,:self.feature_dim]
        feature_3_2 = feature_2 * weight3[:,self.feature_dim:2*self.feature_dim]

        feature  = feature_3_1 + feature_3_2

        return feature 
    
    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#group_num = 128, there are three layers in fusion sub-network, for three models to ensemble
class Combination_three(nn.Module):
    def __init__(self,features_dim:int,num_classes:int):
        super(Combination_three,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes

        hidden_size = 1024

        if (features_dim*num_classes) > 4096:
            assert num_classes > 0 and features_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, 1024)
            self.linear_weight1 = nn.Sequential(
                nn.Linear(3*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*9),
                nn.Sigmoid()
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(3*1024, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*3),
                nn.Sigmoid()
            )
        else:
            self.map = MultiLinearMap()
            self.linear_weight1 = nn.Sequential(
                nn.Linear(3*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.feature_dim*9),
                nn.Sigmoid()
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(3*(self.feature_dim*self.num_classes), hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,self.feature_dim*3),
                nn.Sigmoid()
            )

        self.relu3 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor,feature3:torch.Tensor,label3:torch.Tensor):
        label1 = F.softmax(label1, dim=1)
        h1 = self.map(feature1, label1)

        label2 = F.softmax(label2, dim=1)
        h2 = self.map(feature2, label2)

        label3 = F.softmax(label3, dim=1)
        h3 = self.map(feature3, label2)

        input = torch.cat((h1,h2,h3),dim=1)

        weight1 = self.linear_weight1(input)
    
        weight2 = self.linear_weight2(input)

        #linear1
        feature_1_1 = feature1 * weight1[:,:self.feature_dim]
        feature_1_2 = feature1 * weight1[:,self.feature_dim:2*self.feature_dim]
        feature_1_3 = feature1 * weight1[:,self.feature_dim*2:self.feature_dim*3]
        feature_1_4 = feature2 * weight1[:,self.feature_dim*3:self.feature_dim*4]
        feature_1_5 = feature2 * weight1[:,self.feature_dim*4:self.feature_dim*5]
        feature_1_6 = feature2 * weight1[:,self.feature_dim*5:self.feature_dim*6]
        feature_1_7 = feature3 * weight1[:,self.feature_dim*6:self.feature_dim*7]
        feature_1_8 = feature3 * weight1[:,self.feature_dim*7:self.feature_dim*8]
        feature_1_9 = feature3 * weight1[:,self.feature_dim*8:self.feature_dim*9]

        feature_1 = feature_1_1+feature_1_4 + feature_1_7
        feature_2 = feature_1_2+feature_1_5 + feature_1_8
        feature_3 = feature_1_3+feature_1_6 + feature_1_9

        feature_1 = self.relu3(feature_1)
        feature_2 = self.relu3(feature_2)
        feature_3 = self.relu3(feature_3)

        feature_2_1 = feature_1 * weight2[:,:self.feature_dim]
        feature_2_2 = feature_2 * weight2[:,self.feature_dim:2*self.feature_dim]
        feature_2_3 = feature_3 * weight2[:,self.feature_dim*2:3*self.feature_dim]

        feature  = feature_2_1 + feature_2_2 + feature_2_3

        return feature 
    
    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#the parameters of fusion sub_network is learnt
class Combination_Korn_learn(nn.Module):
    def __init__(self,features_dim:int,num_classes:int):
        super(Combination_Korn_learn,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes

        hidden_size = 1024

        self.weight1 = torch.nn.Parameter(torch.tensor(np.ones((1, self.feature_dim * 4))).float(), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.tensor(np.ones((1, self.feature_dim * 2))).float(), requires_grad=True)
        

        self.relu3 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor):
        #linear1
        feature_1_1 = feature1 * self.weight1[:,:self.feature_dim]
        feature_1_2 = feature1 * self.weight1[:,self.feature_dim:2*self.feature_dim]
        feature_1_3 = feature2 * self.weight1[:,self.feature_dim*2:self.feature_dim*3]
        feature_1_4 = feature2 * self.weight1[:,self.feature_dim*3:self.feature_dim*4]

        feature_1 = feature_1_1+feature_1_3
        feature_2 = feature_1_2+feature_1_4
        
        feature_1 = self.relu3(feature_1)
        feature_2 = self.relu3(feature_2)

        feature_2_1 = feature_1 * self.weight2[:,:self.feature_dim]
        feature_2_2 = feature_2 * self.weight2[:,self.feature_dim:2*self.feature_dim]

        feature  = feature_2_1 + feature_2_2

        return feature 
    
    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params



#use fully connected layer to replace shuffle layer
class Combination_Linear(nn.Module):
    def __init__(self,features_dim:int,num_classes:int):
        super(Combination_Linear,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes

        hidden_size = 1024

        self.linear = nn.Sequential(
            nn.Linear(2*(self.feature_dim), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,self.feature_dim),
            nn.ReLU()
        )

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor):
        
        input = torch.cat((feature1,feature2),dim=1)

        feature = self.linear(input)

        return feature 
    
    def get_parameters(self, base_lr=1.0):
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params


#group is leas than 128
class Combination_tra(nn.Module):
    def __init__(self,features_dim:int,num_classes:int,W,group_num):
        super(Combination_tra,self).__init__()
        self.feature_dim = features_dim
        self.num_classes = num_classes
        self.group_num = group_num
        self.group_dim = int(self.feature_dim/self.group_num)
        self.W =W

        if (features_dim*num_classes) > 4096:
            assert num_classes > 0 and features_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, 1024)
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*1024,  self.group_dim*self.group_dim*4*10),
                nn.ReLU(),
                nn.Linear( self.group_dim*self.group_dim*4*10,  self.group_dim*self.group_dim*4*10),
                nn.ReLU(),
                nn.Linear( self.group_dim*self.group_dim*4*10, self.group_dim*self.group_dim*4),
            )
            self.linear_weight2 = nn.Sequential(
                nn.Linear(2*1024, self.group_dim*self.group_dim*2*10),
                nn.ReLU(),
                nn.Linear(self.group_dim*self.group_dim*2*10, self.group_dim*self.group_dim*2*10),
                nn.ReLU(),
                nn.Linear(self.group_dim*self.group_dim*2*10, self.group_dim*self.group_dim*2),
            )
        else:
            self.map = MultiLinearMap()
            self.linear_weight1 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), self.group_dim*self.group_dim*4*10),
                nn.ReLU(),
                nn.Linear(self.group_dim*self.group_dim*4*10, self.group_dim*self.group_dim*4*10),
                nn.ReLU(),
                nn.Linear(self.group_dim*self.group_dim*4*10, self.group_dim*self.group_dim*4)
            )
            hidden_size = self.group_dim*self.group_dim*2*10
            self.linear_weight2 = nn.Sequential(
                nn.Linear(2*(self.feature_dim*self.num_classes), hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size,self.group_dim*self.group_dim*2),
            )


        self.relu3 = nn.ReLU()
        self.sigmoid3 = nn.ReLU()

    
    def forward(self,feature1:torch.Tensor,label1:torch.Tensor,feature2:torch.Tensor,label2:torch.Tensor):
        label1 = F.softmax(label1, dim=1)
        h1 = self.map(feature1, label1)

        label2 = F.softmax(label2, dim=1)
        h2 = self.map(feature2, label2)

        input = torch.cat((h1,h2),dim=1)

        weight1 = self.linear_weight1(input).view((input.shape[0],self.group_dim*2,self.group_dim*2))

        weight2 = self.linear_weight2(input).view((input.shape[0],self.group_dim,self.group_dim*2))
        #linear1
        f = torch.cat((feature1,feature2),dim=1)
        w =torch.matmul(weight1,self.W[:self.group_dim*2,:])
        feature = torch.bmm(w,f.unsqueeze(2))
        for i in np.arange(self.group_num-1):
            w = torch.matmul(weight1, self.W[(i+1)*self.group_dim*2:(i+2)*self.group_dim*2,:])
            feature_sep = torch.bmm(w,f.unsqueeze(2))
            feature = torch.cat((feature,feature_sep),dim=1)
        
        f = self.relu3(feature)
        #linear2
        feature = torch.bmm(torch.matmul(weight2,self.W[:self.group_dim*2,:]),f)
        for i in np.arange(self.group_num-1):
            w = torch.matmul(weight2, self.W[(i+1)*self.group_dim*2:(i+2)*self.group_dim*2,:])
            feature_sep = torch.bmm(w,f)
            feature = torch.cat((feature,feature_sep),dim=1)
        
        feature = self.sigmoid3(feature.squeeze(2))

        return feature 
    
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.parameters(), "lr": 1.0 * base_lr},
        ]
        return params