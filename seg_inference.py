import random
import os
from argparse import ArgumentParser
from utils.tools import get_config, get_model_list, decode_segmap

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model.network_seg_contour import Parser
from utils.test_dataset import Test_Dataset

import numpy as np
import scipy.misc as misc

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/seg_config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')

def main():
    # Config file reading
    args = parser.parse_args()
    config = get_config(args.config)

    # ------ CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True


    if not os.path.exists(config['output_test_dir']):
        os.makedirs(config['output_test_dir'])

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    # first, read images and pick labels with same name
    # we will train all images from HQ dataset


    # ---------- train and test dataset&loader
    try:  # for unexpected error logging
        # Load the dataset
        print("Inference on dataset: {" + config['dataset_name'] + "}")
        test_dataset = Test_Dataset(data_path=config['test_data_path'],
                                    with_subfolder=config['data_with_subfolder'],
                                    image_shape=config['image_shape'],
                                    random_crop=config['random_crop'], return_name=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False, num_workers=config['num_workers'])

        # [Trainer] (in test, not use trainer class directly)
        netG = Parser(config)

        # Get the resume iteration to restart training
        #================== <LOAD CHECKPOINT FILE starting with parser*.pt> ============================
        last_checkpoint_file = get_model_list(config['resume'], "parser", config['resume_iter'])
        netG.load_state_dict(torch.load(last_checkpoint_file))
        print("Resume from {}".format(config['resume']))

        # CUDA AVAILABLE
        if cuda:
            netG = nn.parallel.DataParallel(netG, device_ids=device_ids)

        # connect loaders to iter()
        iterable_test_loader = iter(test_loader)

        # learing rate
        #lr = config['lr']

        print('Inference Start.........')
        start_iteration = 0

        # =============== TEST ===================
        total_iter = int(len(test_dataset.samples) / config['test_batch']) + 1
        
        #for iteration in range(start_iteration, config['niter'] + 1):
        for iteration in range(total_iter):
            print('ITERATION {}..... [{}/{}]'.format(iteration, iteration*config['test_batch'],
                                                     int(len(test_dataset.samples))))
            try:
                test_img_names, test_orig_images = iterable_test_loader.next()
            except StopIteration:
                iterable_test_loader = iter(test_loader)
                test_img_names, test_orig_images = iterable_test_loader.next()

            if cuda:
                test_orig_images = test_orig_images.cuda()

            # <predict test set>
            test_predict = netG(test_orig_images)

            for test_idx in range(test_orig_images.shape[0]):
                pred_out = torch.argmax(test_predict[test_idx], dim=0)
                test_sam = pred_out.cpu().numpy()

                if config['save_result_as_colored']:
                    decoded = decode_segmap(test_sam)
                    misc.imsave(os.path.join(config['output_test_dir'], test_img_names[test_idx].split('.')[0] + '.png'), decoded)
                else:
                    cv2.imwrite(os.path.join(config['output_test_dir'], test_img_names[test_idx].split('.')[0] + '.png'), test_sam)


    except Exception as e:  # for unexpected error logging (set
        print("{e}")
        raise e




if __name__ == '__main__':
    main()
