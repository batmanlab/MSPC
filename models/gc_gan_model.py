import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# from torch.nn.parallel import DistributedDataParallel as DDP
import random
import math
import sys
import pdb


class GcGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--identity', type=float, default=0.3,
                                 help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. '
                                      'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, '
                                      'please set optidentity = 0.1')
        parser.add_argument('--lambda_AB', type=float, default=10.0, help='weight for gc loss')
        parser.add_argument('--lambda_gc', type=float, default=2.0, help='trade-off parameter for Gc and idt')
        parser.add_argument('--lambda_G', type=float, default=1.0, help='trade-off parameter for G, gc, and idt')
        parser.add_argument('--geometry', type=str, default='rot',
                            help='type of consitency.')
        parser.add_argument('--idt', type=util.str2bool, nargs='?', const=True, default=True,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.name = opt.name
        self.netG_AB = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_AB = DDP(self.netG_AB, broadcast_buffers=False)
        if self.isTrain:
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_gc_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD_B = DDP(self.netD_B, broadcast_buffers=False)
            # self.netD_gc_B = DDP(self.netD_gc_B, broadcast_buffers=False)
        self.loss_names = ['D_B', 'G_AB', 'G_gc_AB']
        self.visual_names = ['real_A', 'fake_B', 'fake_gc_B', 'real_B']

        if opt.idt and self.isTrain:
            self.loss_names += ['idt', 'idt_gc', 'gc']

        if self.isTrain:
            self.model_names = ['G_AB', 'D_B', 'D_gc_B']
        else:  # during test time, only load G
            self.model_names = ['G_AB']

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_gc_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionGc = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters()), lr=opt.lr,
                                                betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(
                itertools.chain(self.netD_B.parameters(), self.netD_gc_B.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def backward_D_basic(self, netD, real, fake, netD_gc, real_gc, fake_gc):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # Real_gc
        pred_real_gc = netD_gc(real_gc)
        loss_D_gc_real = self.criterionGAN(pred_real_gc, True)
        # Fake_gc
        pred_fake_gc = netD_gc(fake_gc.detach())
        loss_D_gc_fake = self.criterionGAN(pred_fake_gc, False)
        # Combined loss
        loss_D += (loss_D_gc_real + loss_D_gc_fake) * 0.5

        # backward
        loss_D.backward()
        return loss_D

    def get_image_paths(self):
        return self.image_paths

    def rot90(self, tensor, direction):
        tensor = tensor.transpose(2, 3)
        size = self.opt.crop_size
        inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
        if direction == 0:
            tensor = torch.index_select(tensor, 3, inv_idx)
        else:
            tensor = torch.index_select(tensor, 2, inv_idx)
        return tensor

    def forward(self):
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        self.real_A = self.input_A
        self.real_B = self.input_B

        size = self.opt.crop_size

        if self.opt.geometry == 'rot':
            self.real_gc_A = self.rot90(input_A, 0)
            self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
            self.real_gc_A = torch.index_select(input_A, 2, inv_idx)
            self.real_gc_B = torch.index_select(input_B, 2, inv_idx)
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

    def get_gc_rot_loss(self, AB, AB_gc, direction):
        loss_gc = 0.0

        if direction == 0:
            AB_gt = self.rot90(AB_gc.clone().detach(), 1)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 0)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)
        else:
            AB_gt = self.rot90(AB_gc.clone().detach(), 0)
            loss_gc = self.criterionGc(AB, AB_gt)
            AB_gc_gt = self.rot90(AB.clone().detach(), 1)
            loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        # loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def get_gc_vf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.crop_size

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 2, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 2, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        # loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def get_gc_hf_loss(self, AB, AB_gc):
        loss_gc = 0.0

        size = self.opt.crop_size

        inv_idx = torch.arange(size - 1, -1, -1).long().cuda()

        AB_gt = torch.index_select(AB_gc.clone().detach(), 3, inv_idx)
        loss_gc = self.criterionGc(AB, AB_gt)

        AB_gc_gt = torch.index_select(AB.clone().detach(), 3, inv_idx)
        loss_gc += self.criterionGc(AB_gc, AB_gc_gt)

        loss_gc = loss_gc * self.opt.lambda_AB * self.opt.lambda_gc
        # loss_gc = loss_gc*self.opt.lambda_AB
        return loss_gc

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_gc_B = self.fake_gc_B_pool.query(self.fake_gc_B)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B, self.netD_gc_B, self.real_gc_B, fake_gc_B)
        self.loss_D_B = loss_D_B.item()

    def backward_G(self):
        # adversariasl loss
        fake_B = self.netG_AB.forward(self.real_A)
        pred_fake = self.netD_B.forward(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        fake_gc_B = self.netG_AB.forward(self.real_gc_A)
        pred_fake = self.netD_gc_B.forward(fake_gc_B)
        loss_G_gc_AB = self.criterionGAN(pred_fake, True) * self.opt.lambda_G

        if self.opt.geometry == 'rot':
            loss_gc = self.get_gc_rot_loss(fake_B, fake_gc_B, 0)
        elif self.opt.geometry == 'vf':
            loss_gc = self.get_gc_vf_loss(fake_B, fake_gc_B)

        if self.opt.identity > 0:
            # G_AB should be identity if real_B is fed.
            idt_A = self.netG_AB(self.real_B)
            loss_idt = self.criterionIdt(idt_A, self.real_B) * self.opt.lambda_AB * self.opt.identity
            idt_gc_A = self.netG_AB(self.real_gc_B)
            loss_idt_gc = self.criterionIdt(idt_gc_A, self.real_gc_B) * self.opt.lambda_AB * self.opt.identity

            self.idt_A = idt_A.data
            self.idt_gc_A = idt_gc_A.data
            self.loss_idt = loss_idt.item()
            self.loss_idt_gc = loss_idt_gc.item()
        else:
            loss_idt = 0
            loss_idt_gc = 0
            self.loss_idt = 0
            self.loss_idt_gc = 0

        loss_G = loss_G_AB + loss_G_gc_AB + loss_gc + loss_idt + loss_idt_gc

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_gc_B = fake_gc_B.data

        self.loss_G_AB = loss_G_AB.item()
        self.loss_G_gc_AB = loss_G_gc_AB.item()
        self.loss_gc = loss_gc.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_AB
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_B and D_gc_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()


    def test(self):
        # self.netG_AB.eval()
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        input_A = self.input_A.clone()
        input_B = self.input_B.clone()

        size = self.opt.crop_size

        if self.opt.geometry == 'rot':
            self.real_gc_A = self.rot90(input_A, 0)
            self.real_gc_B = self.rot90(input_B, 0)
        elif self.opt.geometry == 'vf':
            inv_idx = torch.arange(size - 1, -1, -1).long().cuda()
            self.real_gc_A = Variable(torch.index_select(input_A, 2, inv_idx))
            self.real_gc_B = Variable(torch.index_select(input_B, 2, inv_idx))
        else:
            raise ValueError("Geometry transformation function [%s] not recognized." % self.opt.geometry)

        self.fake_B = self.netG_AB.forward(self.real_A).data
        self.fake_gc_B = self.netG_AB.forward(self.real_gc_A).data