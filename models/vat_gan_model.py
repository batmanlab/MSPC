import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.autograd import grad
from .vat import VAT_pert

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

def on_after_backward(self) -> None:
    valid_gradients = True
    for name, param in self.named_parameters():
        if param.grad is not None:
            valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
            if not valid_gradients:
                break

    if not valid_gradients:
        print(f'detected inf or nan values in gradients. not updating model parameters')
        self.zero_grad()


class VATGANModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--identity', type=float, default=0.3,
                            help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. '
                                 'For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, '
                                 'please set optidentity = 0.1')
        parser.add_argument('--lambda_AB', type=float, default=10.0, help='weight for consistency loss')
        parser.add_argument('--lambda_pert', type=float, default=0.001, help='weight for gradient perturbation')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--idt', type=util.str2bool, nargs='?', const=True, default=True,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.name = opt.name
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']
        self.visual_names = ['real_A', 'fake_B',  'real_B']

        if opt.idt and self.isTrain:
            self.loss_names += ['max_pert', 'idt', 'idt_perturbation']
            self.visual_names += ['idt_B', 'fake_B_perturbation','idt_B_perturbation']

        if self.isTrain:
            self.model_names = ['G', 'D', 'VAT_pert']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.netG, self.opt.norm,
                                        not self.opt.no_dropout, self.opt.init_type, self.opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_perturbation = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netVAT_pert = VAT_pert(eps=opt.lambda_pert)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_perturbation = torch.optim.Adam(self.netD_perturbation.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]

    def optimize_parameters(self):
        # forward
        self.forward()
        self.forward_perturbation()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        # average_gradients(self.netD)
        on_after_backward(self.netD)
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        # average_gradients(self.netG)
        on_after_backward(self.netG)
        self.optimizer_G.step()


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        fake_perturbation = self.fake_B_perturbation.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_perturbation = self.netD_perturbation(fake_perturbation)
        self.loss_D_fake_perturbation = self.criterionGAN(pred_fake_perturbation, False).mean()
        # Real
        self.pred_real_perturbation = self.netD_perturbation(self.real_B)
        self.loss_D_real_perturbation = self.criterionGAN(self.pred_real_perturbation, True).mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.loss_D_fake_perturbation + self.loss_D_real_perturbation) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        fake_perturbation = self.fake_B_perturbation
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            pred_fake_perturbation = self.netD_perturbation(fake_perturbation)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            self.loss_G_GAN_perturbation = self.criterionGAN(pred_fake_perturbation, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_max_pert =  self.criterionIdt(self.fake_B.detach(), self.fake_B_perturbation)
        #
        if self.opt.idt:
            self.loss_idt = self.criterionIdt(self.idt_B, self.real_B)
            self.loss_idt_perturbation = self.criterionIdt(self.idt_B_perturbation, self.real_B)

        loss_G = (self.loss_G_GAN + self.opt.identity*self.opt.lambda_AB*self.loss_idt)*0.5
        loss_G_perturbation = (self.loss_G_GAN_perturbation + self.opt.identity*self.opt.lambda_AB*(self.loss_idt_perturbation+self.loss_max_pert)*0.5)*0.5
        return loss_G + loss_G_perturbation

    def forward_perturbation(self):

        vat_e = self.netVAT_pert(self.netG, self.real)

        self.fake_perturbation = self.netG(self.real + vat_e.detach())
        self.fake_B_perturbation = self.fake_perturbation[:self.real_A.size(0)]

        if self.opt.idt:
            self.idt_B_perturbation = self.fake_perturbation[self.real_A.size(0):]





