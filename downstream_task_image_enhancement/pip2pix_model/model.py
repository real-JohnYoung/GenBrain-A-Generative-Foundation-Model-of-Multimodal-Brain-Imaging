import os
import torch
import torch.nn as nn
import functools
from torch.nn import init


class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        return lambda x: Identity()
    else:
        raise NotImplementedError('Normalization [%s] not found' % norm_type)

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if gpu_ids:
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain)
    return net

class GANLoss(nn.Module):
    def __init__(self, gan_mode, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError()

    def get_target_tensor(self, prediction, is_real):
        return (self.real_label if is_real else self.fake_label).expand_as(prediction)

    def forward(self, prediction, is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target = self.get_target_tensor(prediction, is_real)
            return self.loss(prediction, target)
        elif self.gan_mode == 'wgangp':
            return -prediction.mean() if is_real else prediction.mean()

class UnetSkipConnectionBlock3D(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer.func == nn.InstanceNorm3d if isinstance(norm_layer, functools.partial) else norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, 4, 2, 1, bias=use_bias)
        upconv = nn.ConvTranspose3d(inner_nc * 2 if not innermost else inner_nc, outer_nc, 4, 2, 1, bias=use_bias)
        down = [nn.LeakyReLU(0.2, True), downconv]
        up = [nn.ReLU(True), upconv, norm_layer(outer_nc)]
        if outermost:
            model = [downconv, submodule, nn.ReLU(True), upconv, nn.Tanh()]
        elif innermost:
            model = down + up
        else:
            model = down + [submodule] + up
            if use_dropout:
                model += [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x) if self.outermost else torch.cat([x, self.model(x)], 1)

class Unet3DGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=5, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super().__init__()
        unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, innermost=True, norm_layer=norm_layer)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock3D(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock3D(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock3D(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock3D(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d):
        super().__init__()
        use_bias = norm_layer.func == nn.InstanceNorm3d if isinstance(norm_layer, functools.partial) else norm_layer == nn.InstanceNorm3d
        kw = 4
        padw = 1
        sequence = [nn.Conv3d(input_nc, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]
        sequence += [nn.Conv3d(ndf * nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def define_G(input_nc, output_nc, num_down, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm)
    net = Unet3DGenerator(input_nc, output_nc, num_down, ngf, norm_layer, use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm)
    net = NLayerDiscriminator3D(input_nc, ndf, n_layers_D, norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

class Pix2PixModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.gpu_ids else 'cpu')
        self.isTrain = opt.isTrain
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.num_down, opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        if self.isTrain:
            self.netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

    def forward(self):
        self.fake_B = self.netG(self.real_A)

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()
        return self.loss_D.item()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        return self.loss_G.item()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        loss_D = self.backward_D()
        self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        loss_G = self.backward_G()
        self.optimizer_G.step()
        return loss_D, loss_G
