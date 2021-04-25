from PIL import Image
from dataset import get_loader
import torch
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import os
from models import GICD



# #
# pre_dict = torch.load('./param/checkpoint.pth')
# param = pre_dict['state_dict']
# epoch = pre_dict['epoch']

# print(epoch)

# torch.save(param, './param/gicd.pth')

# ------------------------------------------------ 
# device = torch.device("cuda")
# model = GICD()
# model = model.to(device)
# checkpoint = torch.load('./param/checkpoint.pth')
# pretrained_dict = checkpoint['state_dict']
# model.to(device)



# model_dict=model.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}


# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# model_dict = model.state_dict()

# torch.save(model_dict, './param/gicd.pth')


# ------------------------------------------------ 



device = torch.device("cuda")
model = GICD()
model = model.to(device)
gicd_dic = torch.load('./param/gicd.pth')
model.to(device)
model.load_state_dict(gicd_dic)

ginet_dict = model.ginet.state_dict()

torch.save(ginet_dict, './param/gicd_ginet.pth')