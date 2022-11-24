import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import os
import torchsnooper

class GradCAM(nn.Module):
	def __init__(self, backbone, head=None, device='cuda:0'):
		super(GradCAM, self).__init__()
		self.backbone = backbone
		self.head = head
		self.gradients = []
		self.handle = []
		self.device = device

	def save_gradient(self, grad):
		self.gradients.append(grad)

	# @torchsnooper.snoop()
	def forward(self, images):
		# with torch.no_grad():
		feature = self.backbone.backbone(images)
		handle = feature.register_hook(self.save_gradient)
		self.handle.append(handle)
		f = self.backbone.pool_layer(feature)
		f = self.backbone.bottleneck(f)
		prediction = self.head(f)
		# prediction = self.backbone.head(f)

		return feature.detach(), prediction

	# @torchsnooper.snoop()
	def __call__(self, images, targets, index):
		feature, out = self.forward(images)

		one_hot = F.one_hot(targets, 65).cuda()
		one_hot = one_hot.type(torch.float32).requires_grad_(True)

		one_hot = torch.sum(one_hot * out)

		self.zero_grad()
		one_hot.backward()
		grads_val = self.gradients[0]

		for handle in self.handle:
			handle.remove()

		del one_hot

		weightt = torch.mean(grads_val, dim=(2,3)).unsqueeze(2).unsqueeze(2)
		camm = torch.sum(torch.mul(weightt, feature), dim=1).detach()
		v_cam = cv2.resize(camm.permute(1, 2, 0).cpu().data.numpy(), images.shape[2:]).swapaxes(0, 2).swapaxes(1, 2)
		v_cam -= np.min(v_cam, axis=(1, 2)).reshape(-1, 1, 1)
		v_cam /= np.max(v_cam, axis=(1, 2)).reshape(-1, 1, 1)
		show_cam_on_image(images, targets, v_cam, index)

		self.gradients.clear()
		self.handle.clear()
		return feature, out


def show_cam_on_image(images, targets, cam, index):
	for i in range(images.shape[0]):
		save_dir = f'GradCAM/average_distill/'
		if not os.path.exists(save_dir): os.makedirs(save_dir)
		img = images[i].permute(1, 2, 0).cpu().data.numpy()
		img *= (0.229, 0.224, 0.225)
		img += (0.485, 0.456, 0.406)
		heatmap = cv2.applyColorMap(np.uint8(255 * cam[i]), cv2.COLORMAP_JET)
		heatmap = np.float32(heatmap) / 255
		camm = heatmap + img
		camm = camm / np.max(camm)
		cv2.imwrite(save_dir + f'cam_{i + index}_{targets[i]}.jpg', np.uint8(255 * camm))

if __name__ == '__main__':
	# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	data_root = VOC_ROOT
	valset = VOCDetection(root=data_root,
						  image_sets=[('2007', 'test')],
						  transform=BaseTransform(300, MEANS), vis=True)
	model = torch.load("/remote-home/hanchengye/DLprune/model/Finetune/original/20200407/ssd300_VOC_395000.pkl").cuda()
	val_loader = DataLoader(valset, batch_size=32, num_workers=8, shuffle=False, collate_fn=detection_collate, pin_memory=True)
	layer_index = []
	name_index = {}
	part = ['vgg', 'extras']
	for k, v in model.named_modules():
		if hasattr(v, 'weight') and any([p in k for p in part]):
			layer_index.append(v)
			name_index[k] = v

	vis = GradCAM_SSD300(model, layer_index, name_index)
	# image = torch.randn(2, 3, 300,300).cuda()
	# target = [Variable(torch.Tensor([[0.1, 0.1, 0.5, 0.5, 2],[0.1, 0.3, 0.2, 0.6, 15]])).cuda(),
	# 		  Variable(torch.Tensor([[0.1, 0.5, 0.5, 0.8, 1],[0.1, 0.3, 0.2, 0.6, 19]])).cuda()]
	model.train()
	for img_ids, imgs, targets in val_loader:
		imgs = imgs.cuda()
		T_feature_embedding, out_T, loc_t, conf_t = vis(img_ids, imgs, targets)
	print(conf_t)
