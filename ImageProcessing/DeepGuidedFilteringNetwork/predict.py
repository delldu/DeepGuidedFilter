import os
import argparse

import torch
from torch.autograd import Variable

from tqdm import tqdm

from dataset import PreSuDataset
from utils import tensor_to_img, calc_metric, Config
from module import DeepGuidedFilter, DeepGuidedFilterAdvanced, DeepGuidedFilterConvGF, DeepGuidedFilterGuidedMapConvGF

from skimage.io import imsave

import pdb

parser = argparse.ArgumentParser(description='Predict with Deep Guided Filtering Networks')
parser.add_argument('--task',        type=str, default='auto_ps',                  help='TASK')
parser.add_argument('--img_path',    type=str, default=None,                       help='IMG_PATH')
parser.add_argument('--img_list',    type=str, default=None,                       help='IMG_LIST')
parser.add_argument('--save_folder', type=str, required=True,                      help='SAVE_FOLDER')
parser.add_argument('--model',       type=str, default='deep_guided_filter',       help='model')

parser.add_argument('--low_size',    type=int, default=64,                         help='LOW_SIZE')
parser.add_argument('--gpu',         type=int, default=0,                          help='GPU')
parser.add_argument('--gray',                  default=False, action='store_true', help='GPU')
args = parser.parse_args()

# Test Images
img_list = []
if args.img_path is not None:
    img_list.append(args.img_path)
if args.img_list is not None:
    with open(args.img_list) as f:
        for line in f:
            img_list.append(line.strip())
assert len(img_list) > 0

# Save Folder
if not os.path.isdir(args.save_folder):
    os.makedirs(args.save_folder)

# Model
# args.model -- 'deep_guided_filter_advanced'

if args.model in ['guided_filter', 'deep_guided_filter']:
    model = DeepGuidedFilter()
elif args.model == 'deep_guided_filter_advanced': # True
    model = DeepGuidedFilterAdvanced()
elif args.model == 'deep_conv_guided_filter':
    model = DeepGuidedFilterConvGF()
elif args.model == 'deep_conv_guided_filter_adv':
    model = DeepGuidedFilterGuidedMapConvGF()
else:
    print('Not a valid model!')
    exit(-1)

model2name = {'guided_filter': 'lr',
              'deep_guided_filter': 'hr',
              'deep_guided_filter_advanced': 'hr_ad', # True
              'deep_conv_guided_filter': 'conv_hr',
              'deep_conv_guided_filter_adv': 'conv_hr_ad'}
model_path = os.path.join('models', args.task, '{}_net_latest.pth'.format(model2name[args.model]))

# pp args.task -- 'auto_ps'

if args.model == 'guided_filter':
    model.init_lr(model_path)
else:
    model.load_state_dict(torch.load(model_path))

# model_path -- 'models/auto_ps/hr_ad_net_latest.pth'
# img_list -- ['../../images/auto_ps.jpg']

# data set
test_data = PreSuDataset(img_list, low_size=args.low_size)
# pp args.low_size -- 64

# GPU
if args.gpu >= 0:
    with torch.cuda.device(args.gpu):
        model.cuda()

# test
i_bar = tqdm(total=len(test_data), desc='#Images')
for idx, imgs in enumerate(test_data):
    name = os.path.basename(test_data.get_path(idx))

    lr_x, hr_x = imgs[1].unsqueeze(0), imgs[0].unsqueeze(0)
    if args.gpu >= 0:
        with torch.cuda.device(args.gpu):
            lr_x = lr_x.cuda()
            hr_x = hr_x.cuda()

    # (Pdb) pp lr_x.size()
    # torch.Size([1, 3, 96, 64])
    # (Pdb) pp hr_x.size()
    # torch.Size([1, 3, 1536, 1024])

    # imgs = model(Variable(lr_x), Variable(hr_x)).data.cpu()

    imgs = model(lr_x, hr_x).data.cpu()
    # (Pdb) imgs.size()
    # torch.Size([1, 3, 1536, 1024])

    for img in imgs:
        img = tensor_to_img(img, transpose=True)
        if args.gray:
            img = img.mean(axis=2).astype(img.dtype)
        imsave(os.path.join(args.save_folder, name), img)

    i_bar.update()