import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
 
Resnet = models.resnet50(pretrained=True)

#resnet  conv1 and conv2
class Part1(nn.Module):
    def __init__(self):
        super(Part1,self).__init__()
        self.part = nn.Sequential(*list(Resnet.children())[:5])
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
    def forward(self,x:torch.Tensor):
        out = self.part(x)
        #计算相似度的out
        out_sim = self.pool_layer(out)

        return out,out_sim
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part.parameters(), "lr": 1.0 * base_lr},
        ]
        return params


#conv3
class Part2(nn.Module):
    def __init__(self):
        super(Part2,self).__init__()
        self.part = nn.Sequential(*list(Resnet.children())[5])
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
    def forward(self,x:torch.Tensor):
        out = self.part(x)
        #计算相似度的out
        out_sim = self.pool_layer(out)

        return out,out_sim
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#conv4
class Part3(nn.Module):
    def __init__(self):
        super(Part3,self).__init__()
        self.part = nn.Sequential(*list(Resnet.children())[6])
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
    def forward(self,x:torch.Tensor):
        out = self.part(x)
        #计算相似度的out
        out_sim = self.pool_layer(out)

        return out,out_sim
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#conv5
class Part4(nn.Module):
    def __init__(self):
        super(Part4,self).__init__()
        self.part1 = nn.Sequential(*list(Resnet.children())[7])
        self.part2 = list(Resnet.children())[8]
        self.part3 = list(Resnet.children())[9]
    def forward(self,x:torch.Tensor):
        out = self.part1(x)
        out = self.part2(out).squeeze(3).squeeze(2)
        out = self.part3(out).unsqueeze(2).unsqueeze(3)
        #计算相似度的out
        return out,out
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part1.parameters(), "lr": 1.0 * base_lr},
            {"params": self.part2.parameters(), "lr": 1.0 * base_lr},
            {"params": self.part3.parameters(), "lr": 1.0 * base_lr},
        ]
        return params


#bottleneck and classifier
class Part5(nn.Module):
    def __init__(self,num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
        super(Part5,self).__init__()
    
        self.num_classes = num_classes

        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(1000, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )

        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        if bottleneck is None:
            self.bottleneck = nn.Identity()
            self._features_dim = 1000
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.softmax = torch.nn.Softmax(dim=1)
        self.finetune = finetune

    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.pool_layer(x)
        f = self.bottleneck(f)
        predictions = self.head(f)
        predictions = self.softmax(predictions)
        
        return predictions, f
        

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

#resnet  conv1 and conv2
class Part1_(nn.Module):
    def __init__(self):
        super(Part1_,self).__init__()
        self.part = nn.Sequential(*list(Resnet.children())[:5])
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
    def forward(self,x:torch.Tensor):
        out = self.part(x)
        #计算相似度的out
        with torch.no_grad():
            out_sim = self.pool_layer(out)

        return out,out_sim
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part.parameters(), "lr": 1.0 * base_lr},
        ]
        return params


#conv3
class Part2_(nn.Module):
    def __init__(self):
        super(Part2_,self).__init__()
        self.part = nn.Sequential(*list(Resnet.children())[5])
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
    def forward(self,x:torch.Tensor):
        out = self.part(x)
        #计算相似度的out
        with torch.no_grad():
            out_sim = self.pool_layer(out)

        return out,out_sim
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part.parameters(), "lr": 1.0 * base_lr},
        ]
        return params

#conv4
class Part3_(nn.Module):
    def __init__(self):
        super(Part3_,self).__init__()
        self.part = nn.Sequential(*list(Resnet.children())[6])
        self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
    def forward(self,x:torch.Tensor):
        out = self.part(x)
        #计算相似度的out
        with torch.no_grad():
            out_sim = self.pool_layer(out)

        return out,out_sim
    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        params = [
            {"params": self.part.parameters(), "lr": 1.0 * base_lr},
        ]
        return params



# #res 1后的
# class Part4(nn.Module):
#     def __init__(self,num_classes: int, bottleneck: Optional[nn.Module] = None,
#                  bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
#         super(Part4,self).__init__()
#         self.part1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.part2 = nn.Linear(in_features=256, out_features=1000, bias=True)
    
#         self.num_classes = num_classes

#         bottleneck = nn.Sequential(
#             # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             # nn.Flatten(),
#             nn.Linear(self.part2.out_features, bottleneck_dim),
#             nn.BatchNorm1d(bottleneck_dim),
#             nn.ReLU()
#         )

#         if pool_layer is None:
#             self.pool_layer = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#                 nn.Flatten()
#             )
#         else:
#             self.pool_layer = pool_layer
#         if bottleneck is None:
#             self.bottleneck = nn.Identity()
#             self._features_dim = self.part2.out_features
#         else:
#             self.bottleneck = bottleneck
#             assert bottleneck_dim > 0
#             self._features_dim = bottleneck_dim

#         if head is None:
#             self.head = nn.Linear(self._features_dim, num_classes)
#         else:
#             self.head = head
#         self.finetune = finetune

#     def features_dim(self) -> int:
#         """The dimension of features before the final `head` layer"""
#         return self._features_dim

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """"""
#         f = self.part1(x).squeeze(3).squeeze(2)
#         f = self.part2(f).unsqueeze(2).unsqueeze(3)
#         f = self.pool_layer(f)
#         f = self.bottleneck(f)
#         predictions = self.head(f)
        
#         return predictions, f
        

#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         """A parameter list which decides optimization hyper-parameters,
#             such as the relative learning rate of each layer
#         """
#         params = [
#             {"params": self.part1.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.part2.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
#             {"params": self.head.parameters(), "lr": 1.0 * base_lr},
#         ]

#         return params


# #res2后的
# class Part5(nn.Module):
#     def __init__(self,num_classes: int, bottleneck: Optional[nn.Module] = None,
#                  bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
#         super(Part5,self).__init__()
#         self.part1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.part2 = nn.Linear(in_features=1024, out_features=1000, bias=True)
    
#         self.num_classes = num_classes

#         bottleneck = nn.Sequential(
#             # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             # nn.Flatten(),
#             nn.Linear(self.part2.out_features, bottleneck_dim),
#             nn.BatchNorm1d(bottleneck_dim),
#             nn.ReLU()
#         )

#         if pool_layer is None:
#             self.pool_layer = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#                 nn.Flatten()
#             )
#         else:
#             self.pool_layer = pool_layer
#         if bottleneck is None:
#             self.bottleneck = nn.Identity()
#             self._features_dim = self.part2.out_features
#         else:
#             self.bottleneck = bottleneck
#             assert bottleneck_dim > 0
#             self._features_dim = bottleneck_dim

#         if head is None:
#             self.head = nn.Linear(self._features_dim, num_classes)
#         else:
#             self.head = head
#         self.finetune = finetune

#     def features_dim(self) -> int:
#         """The dimension of features before the final `head` layer"""
#         return self._features_dim

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """"""
#         f = self.part1(x).squeeze(3).squeeze(2)
#         f = self.part2(f).unsqueeze(2).unsqueeze(3)
#         f = self.pool_layer(f)
#         f = self.bottleneck(f)
#         predictions = self.head(f)
        
#         return predictions, f
        

#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         """A parameter list which decides optimization hyper-parameters,
#             such as the relative learning rate of each layer
#         """
#         params = [
#             {"params": self.part1.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.part2.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
#             {"params": self.head.parameters(), "lr": 1.0 * base_lr},
#         ]

#         return params











# import torchvision.models as models
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple, Optional, List, Dict
 
# Resnet = models.resnet50(pretrained=True)

# # part1 = nn.Sequential(*list(list(classifier.children())[0].children())[:4])   #share
# #     part2 = nn.Sequential(*list(list(classifier.children())[0].children())[4][0:2]) #share
# #     part3 = list(list(classifier.children())[0].children())[4][2] #private
# #     part4 = nn.Sequential(*list(list(classifier.children())[0].children())[5][0:3]) #share
# #     part5 = list(list(classifier.children())[0].children())[5][3]  #private
# #     part6 = nn.Sequential(*list(list(classifier.children())[0].children())[6][0:5]) #share
# #     part7 = list(list(classifier.children())[0].children())[6][5] #private
# #     part8 = nn.Sequential(*list(list(classifier.children())[0].children())[7][0:2])   #share
# #     part9 = list(list(classifier.children())[0].children())[7][2] #private
# #     part10 = nn.Sequential(*list(list(classifier.children())[0].children())[8:10]) #share
# #     part11 = nn.Sequential(*list(classifier.children())[1:3]) #share
# class Part1(nn.Module):
#     def __init__(self):
#         super(Part1,self).__init__()
#         self.part = nn.Sequential(*list(Resnet.children())[:4])
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params



# class Part2(nn.Module):
#     def __init__(self):
#         super(Part2,self).__init__()
#         self.part = nn.Sequential(*list(Resnet.children())[4][0:2])
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params

# class Part3(nn.Module):
#     def __init__(self):
#         super(Part3,self).__init__()
#         self.part = list(Resnet.children())[4][2]
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params

# class Part4(nn.Module):
#     def __init__(self):
#         super(Part4,self).__init__()
#         self.part = nn.Sequential(*list(Resnet.children())[5][0:3])
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params

# class Part5(nn.Module):
#     def __init__(self):
#         super(Part5,self).__init__()
#         self.part = list(Resnet.children())[5][3]
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params

# class Part6(nn.Module):
#     def __init__(self):
#         super(Part6,self).__init__()
#         self.part = nn.Sequential(*list(Resnet.children())[6][0:5])
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params

# class Part7(nn.Module):
#     def __init__(self):
#         super(Part7,self).__init__()
#         self.part = list(Resnet.children())[6][5]
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params
    
# class Part8(nn.Module):
#     def __init__(self):
#         super(Part8,self).__init__()
#         self.part = nn.Sequential(*list(Resnet.children())[7][0:2])
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params

# class Part9(nn.Module):
#     def __init__(self):
#         super(Part9,self).__init__()
#         self.part = list(Resnet.children())[7][2]
#     def forward(self,x:torch.Tensor):
#         out = self.part(x)
#         return out
#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         params = [
#             {"params": self.parameters(), "lr": 1.0 * base_lr},
#         ]
#         return params



# class Part10(nn.Module):
#     def __init__(self,num_classes: int, bottleneck: Optional[nn.Module] = None,
#                  bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True, pool_layer=None):
#         super(Part10,self).__init__()
#         self.part1 = list(Resnet.children())[8]
#         self.part2 = list(Resnet.children())[9]
    
#         self.num_classes = num_classes

#         bottleneck = nn.Sequential(
#             # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#             # nn.Flatten(),
#             nn.Linear(self.part2.out_features, bottleneck_dim),
#             nn.BatchNorm1d(bottleneck_dim),
#             nn.ReLU()
#         )

#         if pool_layer is None:
#             self.pool_layer = nn.Sequential(
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1)),
#                 nn.Flatten()
#             )
#         else:
#             self.pool_layer = pool_layer
#         if bottleneck is None:
#             self.bottleneck = nn.Identity()
#             self._features_dim = self.part2.out_features
#         else:
#             self.bottleneck = bottleneck
#             assert bottleneck_dim > 0
#             self._features_dim = bottleneck_dim

        

#         # if head is None:
#         #     self.head = nn.Linear(self._features_dim, num_classes)
#         # else:
#         #     self.head = head
#         self.finetune = finetune

#     def features_dim(self) -> int:
#         """The dimension of features before the final `head` layer"""
#         return self._features_dim

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """"""
        
#         f = self.part1(x).squeeze(2).squeeze(2)
#         f = self.part2(f).unsqueeze(2).unsqueeze(3)
#         f = self.pool_layer(f)
#         f = self.bottleneck(f)
#         # predictions = self.head(f)
#         # if self.training:
#         #     return predictions, f
#         # else:
#         #     return predictions
#         return f

#     def get_parameters(self, base_lr=1.0) -> List[Dict]:
#         """A parameter list which decides optimization hyper-parameters,
#             such as the relative learning rate of each layer
#         """
#         params = [
#             {"params": self.part1.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.part2.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
#             {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
#             # {"params": self.head.parameters(), "lr": 1.0 * base_lr},
#         ]

#         return params
