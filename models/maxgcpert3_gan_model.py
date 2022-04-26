import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import util.util as util
import torch.distributed as dist
from .gc_pert_utils import BoundedGridLocNet, UnBoundedGridLocNet, TPSGridGen, grid_sample
import itertools
import torch.nn.functional as F
from util.image_pool import ImagePool
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.autograd import grad

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
                print(self)
                break

    if not valid_gradients:
        print(f'detected inf or nan values in gradients. not updating model parameters')
        self.zero_grad()

def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z

class Maxgcpert3GANModel(BaseModel):
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
        parser.add_argument('--bounded', type=str, default='bounded', help='weight for consistency loss')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        parser.add_argument('--idt', type=util.str2bool, nargs='?', const=True, default=True,
                            help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--span_range', type=float, default=0.9)
        parser.add_argument('--lambda_blank', type=float, default=1.0)
        parser.add_argument('--grid_size', type=int, default=2)
        parser.add_argument('--pert_threshold', type=float, default=0.1)
        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.opt = opt

        self.name = opt.name + '_' + str(opt.grid_size) + '_' + str(opt.pert_threshold) + '_' + str(
            opt.lambda_blank) + '_' + opt.bounded + '_' + str(opt.identity)
        self.loss_names = ['D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if opt.idt and self.isTrain:
            self.loss_names += ['D_real_perturbation', 'D_fake_perturbation', 'max_pert', 'pert_constraint_D', 'idt',
                                'idt_perturbation']
            self.visual_names += ['fake_B_perturbation', 'fake_B_grid', 'idt_B', 'pert_B', 'idt_B_perturbation',
                                  'pert_A']

        if self.isTrain:
            self.model_names = ['G', 'D', 'D_perturbation', 'LOC', 'TPS']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:

            self.fake_pool = ImagePool(opt.pool_size)
            self.fake_pert_pool = ImagePool(opt.pool_size)

            r1 = self.opt.span_range
            r2 = self.opt.span_range
            assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
            target_control_points = torch.Tensor(list(itertools.product(
                np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (self.opt.grid_size - 1)),
                np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (self.opt.grid_size - 1)),
            )))
            Y, X = target_control_points.split(1, dim=1)
            self.target_control_point = torch.cat([X, Y], dim=1).cuda()


            GridLocNet = {
                'unbounded': UnBoundedGridLocNet,
                'bounded': BoundedGridLocNet,
            }[self.opt.bounded]
            # rotated_point = torch.mm(self.target_control_point, torch.tensor([[0, 1.0], [-1.0, 0]]).t().cuda().float())
            self.netLOC = GridLocNet(self.opt.grid_size, self.opt.grid_size, self.target_control_point)
            self.netTPS = TPSGridGen()

            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_perturbation = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Pert = torch.optim.Adam(itertools.chain(self.netLOC.parameters(), self.netTPS.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netD_perturbation.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_Pert)
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
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netD_perturbation, True)
        self.set_requires_grad(self.netLOC, True)
        self.set_requires_grad(self.netTPS, True)
        self.forward()
        self.forward_perturbation()

        self.set_requires_grad(self.netG, False)
        self.fake_B_perturbation = self.netG(self.pert_A)
        self.fake_B_grid = grid_sample(self.fake_B.detach(), self.grid_A)
        # update D
        self.optimizer_D.zero_grad()
        self.optimizer_Pert.zero_grad()
        self.loss_D, self.loss_pert_D = self.compute_D_loss()
        (self.loss_D + self.loss_pert_D-self.loss_pert_constraint_D*self.opt.lambda_blank).backward()
        self.optimizer_D.step()
        self.optimizer_Pert.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_perturbation, False)
        self.set_requires_grad(self.netLOC, False)
        self.set_requires_grad(self.netTPS, False)
        self.set_requires_grad(self.netG, True)

        self.fake_B_perturbation = self.netG(self.pert_A.detach())
        self.fake_B_grid = grid_sample(self.fake_B, self.grid_A.detach())
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
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
        fake = self.fake_B.detach()#self.fake_pool.query(self.fake_B.detach())
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        self.loss_D_real = self.criterionGAN(self.pred_real, True).mean()

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_perturbation = self.netD_perturbation(self.fake_B_perturbation)
        self.loss_D_fake_perturbation = self.criterionGAN(pred_fake_perturbation, False).mean()
        # Real
        self.pred_real_perturbation = self.netD_perturbation(self.pert_B)
        self.loss_D_real_perturbation = self.criterionGAN(self.pred_real_perturbation, True).mean()

        self.loss_max_pert_D = self.criterionIdt(self.fake_B_perturbation, self.fake_B_grid)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real + self.loss_D_fake_perturbation + self.loss_D_real_perturbation) * 0.5
        self.loss_pert_D = self.opt.identity * self.opt.lambda_AB * self.loss_max_pert_D*0.5

        return self.loss_D, self.loss_pert_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            pred_fake_perturbation = self.netD_perturbation(self.fake_B_perturbation)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            self.loss_G_GAN_perturbation = self.criterionGAN(pred_fake_perturbation, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        self.loss_max_pert = self.criterionIdt(self.fake_B_perturbation,self.fake_B_grid)
        #
        if self.opt.idt:
            self.loss_idt = self.criterionIdt(self.idt_B, self.real_B)
            self.loss_idt_perturbation = self.criterionIdt(self.idt_B_perturbation, self.pert_B.detach())


        loss_G = (self.loss_G_GAN + self.opt.identity*self.opt.lambda_AB*self.loss_idt)*0.5
        loss_G_perturbation = (self.loss_G_GAN_perturbation +
                               self.opt.identity*self.opt.lambda_AB*(self.loss_idt_perturbation + self.loss_max_pert))*0.5
        return loss_G + loss_G_perturbation

    def scale_constraint(self, source_control_points, target_control_points):
        constraint_index = np.random.choice(source_control_points.shape[1], 3, replace=False)
        constraint_source_points1 = source_control_points[:, constraint_index[:2], :]
        constraint_target_points1 = target_control_points[:, constraint_index[:2], :]
        constraint_dis1 = ((constraint_source_points1[:, 0, :] - constraint_source_points1[:, 1, :]) ** 2).sum(
            1).sqrt()
        constraint_dis_t1 = ((constraint_target_points1[:, 0, :] - constraint_target_points1[:, 1, :]) ** 2).sum(
            1).sqrt()

        constraint_source_points2 = source_control_points[:, constraint_index[1:], :]
        constraint_target_points2 = target_control_points[:, constraint_index[1:], :]
        constraint_dis2 = ((constraint_source_points2[:, 0, :] - constraint_source_points2[:, 1, :]) ** 2).sum(
            1).sqrt()
        constraint_dis_t2 = ((constraint_target_points2[:, 0, :] - constraint_target_points2[:, 1, :]) ** 2).sum(
            1).sqrt()

        z = self.opt.pert_threshold
        a = -(z + 1 / z) / 2.0
        b = ((z - 1 / z) / 2.0)

        z = 2.0
        c = -(z + 1 / z) / 2.0
        d = ((z - 1 / z) / 2.0)

        constraint = ((((constraint_dis1 + constraint_dis2) / (constraint_dis_t1 + constraint_dis_t2)) + a).abs()).clamp(min=b).mean() \
                   + ((constraint_dis1 / constraint_dis_t1) / (constraint_dis2 / constraint_dis_t2) + c).abs().clamp(min=d).mean()

        return constraint

    def forward_perturbation(self,):

        batch_size = self.real_A.size(0)

        self.target_control_points = torch.cat([self.target_control_point.unsqueeze(dim=0)] * batch_size,
                                               dim=0)

        ############pertA
        downsample_A = F.interpolate(self.real_A,(64,64),mode='bilinear', align_corners=True)
        source_control_points = self.netLOC(downsample_A)
        source_coordinate = self.netTPS(source_control_points, self.target_control_points, self.opt.crop_size,
                                     self.opt.crop_size)
        self.grid_A = source_coordinate.view(batch_size, self.opt.crop_size, self.opt.crop_size, 2)

        self.pert_A = grid_sample(self.real_A, self.grid_A)

        self.constraint_A = self.scale_constraint(source_control_points, self.target_control_points)
        self.cordinate_contraint_A = ((source_coordinate.mean(dim=1).abs()).clamp(min=0.25)).mean()

        #############pert B
        downsample_B = F.interpolate(self.real_B, (64, 64), mode='bilinear', align_corners=True)
        source_control_points_B = self.netLOC(downsample_B)
        # print(source_control_points[0,:], source_control_points[1,:])
        source_coordinate_B = self.netTPS(source_control_points_B, self.target_control_points, self.opt.crop_size,
                                        self.opt.crop_size)
        self.grid_B = source_coordinate_B.view(batch_size, self.opt.crop_size, self.opt.crop_size, 2)
        self.pert_B = grid_sample(self.real_B, self.grid_B)
        self.idt_B_perturbation = self.netG(self.pert_B.detach())

        self.constraint_B = self.scale_constraint(source_control_points_B, self.target_control_points)
        self.cordinate_contraint_B = ((source_coordinate_B.mean(dim=1).abs()).clamp(min=0.25)).mean()

        self.loss_pert_constraint_D = (self.constraint_A + self.cordinate_contraint_A + self.constraint_B + self.cordinate_contraint_B)*0.5

