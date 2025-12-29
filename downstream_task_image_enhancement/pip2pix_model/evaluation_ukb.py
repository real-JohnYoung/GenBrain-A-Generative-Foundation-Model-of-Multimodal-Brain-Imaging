import os
import torch
import numpy as np
import pandas as pd
from model import Pix2PixModel
from collections import OrderedDict
from tqdm import tqdm

ckpt_path = '/pix2pix_model/exp_001/checkpoints/ckpt_0050000.pt'
save_dir = '/pix2pix_model/ukb_results_001'
cor_data_path = '/corrupt_more_3d_dataset/corrupt'
data_file_path = '../labels/image_quality_enhancement_testing_data.csv'

mni_96 = np.load('../mni_96.npy')
mni_96_nonzero = (mni_96!=0)

data_file =pd.read_csv(data_file_path)


def load_corrupt_images(cor_data_path,eid_batch,modality_batch,device):
    sub_num = len(eid_batch)
    cor_images = []

    for i in range(sub_num):
        modality = modality_batch[i]
        if modality == 11:
            modality_name = 'T1_brain_nonlinear_2mm'
        elif modality == 13:
            modality_name = 'T2_FLAIR_brain_to_MNI'
        cor_image_path = os.path.join(cor_data_path,str(eid_batch[i]),modality_name+'.npy')
        cor_image = np.load(cor_image_path)
        cor_mean = cor_image[mni_96_nonzero].mean()
        cor_std = cor_image[mni_96_nonzero].std()
        cor_image = (cor_image - cor_mean)/cor_std              # attention !!!
        cor_image = torch.tensor(cor_image, dtype=torch.float32,device=device).unsqueeze(0)
        cor_images.append(cor_image)
    
    cor_images = torch.cat(cor_images, dim = 0).unsqueeze(1)
    return cor_images



def save_batch_data(eid_batch,modality_batch, data, save_dir):
    sub_num = len(eid_batch)
    for i in range(sub_num):
        addr = os.path.join(save_dir, f"enhanced_{str(eid_batch[i])}")
        os.makedirs(addr, exist_ok=True)
        # Correct slicing with colon
        x = data[i][0][mni_96_nonzero]
        index = modality_batch[i]
        if index == 11:
            modality = 'T1_brain_nonlinear_2mm'
        elif index ==13:
            modality = 'T2_FLAIR_brain_to_MNI'
        np.save(os.path.join(addr, f"{modality}.npy"), x)



class Opt:
    def __init__(self, gpu_id):
        self.gpu_ids = [gpu_id]
        self.isTrain = True
        self.input_nc = 1
        self.output_nc = 1
        self.ngf = 64
        self.num_down =5
        self.netG = "unet_256"
        self.netD = "basic"
        self.n_layers_D = 3
        self.ndf = 64
        self.norm = "batch"
        self.init_type = "normal"
        self.init_gain = 0.02
        self.no_dropout = False
        self.gan_mode = "lsgan"
        self.lr = 1e-4
        self.beta1 = 0.5
        self.lambda_L1 = 100.0
        self.direction = "AtoB"


opt = Opt(0)
model = Pix2PixModel(opt)

state_dict = torch.load(ckpt_path)
new_state_dict = OrderedDict()
# for k, v in state_dict['model'].items():
#     name = k.replace('module.', '')
#     new_state_dict[name] = v
model.load_state_dict(state_dict['model'])

eid_list = data_file['eid'].astype(str).tolist()
mod_list = data_file['modality'].astype(int).tolist()
sub_num = len(eid_list)

batch_size = 1

for i in tqdm(range(0, sub_num, batch_size)):
    eid_batch = eid_list[i:i+batch_size]
    mod_batch = mod_list[i:i+batch_size]
    data = load_corrupt_images(cor_data_path,eid_batch,mod_batch,device='cuda:0')
    enhanced_data = model.netG(data)
    enhanced_data = enhanced_data.detach().cpu()
    save_batch_data(eid_batch,mod_batch, enhanced_data, save_dir)
