"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch
import numpy as np
import ntpath
import torch.distributed as dist
import torch
from cal_fid import get_fid
import torchvision
from evaluation.evaluate_maps import eval_maps
from evaluation.evaluate_city2parsing import eval_city2parsing
from evaluation.parsing2city.evaluate import eval_parsing2city

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    opt.name = os.path.join(opt.dataroot.strip('../data/'), opt.model,
                            str(opt.batch_size) + '_' + str(opt.crop_size)
                            + '_' + opt.direction
                            + ('_' + opt.netG + '_' + opt.netD))
    # hard-code some parameters for test
    opt.num_threads = 10   # test code only supports num_threads = 1
    opt.batch_size = 20    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.load_size = opt.crop_size
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    opt.name = model.name
    model = create_model(opt)
    model.parallelize()
    model.setup(opt)
    if opt.eval:
        model.eval()
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths

            if i % 5 == 0:  # save images to an HTML file
                print('processing (%04d)-th image... %s' % (i, img_path))

            text_path = os.path.join(save_images(webpage, visuals, img_path, width=opt.display_winsize))
            if opt.model =='cycle_gan':
                save_path_B = os.path.join(text_path, 'fake_B')
                save_path_A = os.path.join(text_path, 'fake_A')
            else:
                save_path = os.path.join(text_path,'fake_B')
            # print(img_path[0])
            for k, path in enumerate(img_path):
                short_path = ntpath.basename(path)
                name = os.path.splitext(short_path)[0]
                if 'T1' in opt.dataroot:
                    np.save(os.path.join(save_path, name),
                        visuals['fake_B'].cpu().numpy()[k])
                else:
                    if opt.model == 'cycle_gan':
                        saved_img = ((((visuals['fake_B'].cpu()[k]) + 1.0) / 2.0))
                        torchvision.utils.save_image(saved_img, os.path.join(save_path_B, name + '.png'))
                        saved_img = ((((visuals['fake_A'].cpu()[k]) + 1.0) / 2.0))
                        torchvision.utils.save_image(saved_img, os.path.join(save_path_A, name + '.png'))
                    else:
                        saved_img = ((((visuals['fake_B'].cpu()[k]) + 1.0) / 2.0))
                        torchvision.utils.save_image(saved_img, os.path.join(save_path, name + '.png'))
                    # util.save_image(saved_img, os.path.join(save_path, name, 'png'))
        webpage.save()  # save the HTML
    if opt.dataroot.strip('../data/') == 'cityscapes':
        if opt.model == 'cycle_gan':
            real_root = os.path.join(opt.dataroot, 'testB')
            metric = eval_city2parsing(real_root, save_path_B)
            real_root = os.path.join(opt.dataroot, 'testB')
            metric += '\n' + eval_parsing2city(real_root, save_path_A)
        else:
            if opt.direction == 'AtoB':
                real_root = os.path.join(opt.dataroot, 'testB')
                metric = eval_city2parsing(real_root, save_path)
            elif opt.direction == 'BtoA':
                real_root = os.path.join(opt.dataroot, 'testB')
                metric = eval_parsing2city(real_root, save_path)

    elif opt.dataroot.strip('../data/') == 'maps':
        if opt.model == 'cycle_gan':
            real_root = os.path.join(opt.dataroot, 'testB')
            metric = eval_maps(real_root, save_path_B)
            real_root = os.path.join(opt.dataroot,  'testA')
            metric += '\n' + eval_maps(real_root, save_path_A)
        else:
            real_root = os.path.join(opt.dataroot, ('testB' if opt.direction == 'AtoB' else 'testA'))
            metric = eval_maps(real_root, save_path)
    else:
        if opt.phase == 'test':
            if opt.model == 'cycle_gan':
                real_root_A = os.path.join(opt.dataroot, 'testA')
                real_root_B = os.path.join(opt.dataroot, 'testB')
            else:
                real_root = os.path.join(opt.dataroot, ('testB' if opt.direction == 'AtoB' else 'testA'))
        elif opt.phase == 'train':
            if opt.model == 'cycle_gan':
                real_root_A = os.path.join(opt.dataroot, 'trainA')
                real_root_B = os.path.join(opt.dataroot, 'trainB')
            else:
                real_root = os.path.join(opt.dataroot, ('trainB' if opt.direction == 'AtoB' else 'trainA'))
        if opt.model == 'cycle_gan':
            fid = get_fid([real_root_A, save_path_A], 50, 2048, 8)
            metric = 'fid: ' + str(fid)
            fid = get_fid([real_root_B, save_path_B], 50, 2048, 8)
            metric += '\n' + 'fid: ' + str(fid)
        else:
            fid = get_fid([real_root, save_path], 50, 2048, 8)
            metric = 'fid: ' + str(fid)
    file = open(os.path.join(text_path, 'eval_result.txt'), 'w')
    file.writelines(metric)