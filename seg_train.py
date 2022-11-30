import random
import time
import shutil
import os
from argparse import ArgumentParser
from utils.tools import get_config, decode_segmap

import cv2
from seg_trainer import Seg_Trainer
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.logger import get_logger
from utils.dataset import Parse_Dataset
from utils.test_dataset import Test_Dataset

import numpy as np
import scipy.misc as misc

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='./seg_config.yaml',
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

    # ----- Directory to save checkpoint file
    checkpoint_path = os.path.join(config['checkpoint_path'])

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call
    if not os.path.exists(config['output_test_dir']):
        os.makedirs(config['output_test_dir'])


    logger.info(f"Arguments: {args}")
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info(f"Random seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)


    logger.info(f"configuration : {config}")

    # first, read images and pick labels with same name
    # we will train all images from HQ dataset


    # ---------- train and test dataset&loader
    try:  # for unexpected error logging
        # Load the dataset
        logger.info(f"Training on dataset: {config['dataset_name']}")
        train_dataset = Parse_Dataset(data_path=config['all_data_path'],
                                      gt_path=config['gt_data_path'],
                                with_subfolder=config['data_with_subfolder'],
                                image_shape=config['image_shape'],
                                random_crop=config['random_crop'], return_name=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=config['batch_size'],
                                                   shuffle=True,
                                                   num_workers=config['num_workers'])

        test_dataset = Test_Dataset(data_path=config['test_data_path'],
                                    with_subfolder=config['data_with_subfolder'],
                                    image_shape=config['image_shape'],
                                    random_crop=config['random_crop'], return_name=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False, num_workers=config['num_workers'])

        # [Trainer]
        trainer = Seg_Trainer(config)
        logger.info(f"\n{trainer.netParser}")


        # CUDA AVAILABLE
        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        # Get the resume iteration to restart training
        #start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1
        start_iteration = trainer_module.resume(config['resume'], config['resume_iter']) if config['resume'] else 1


        # connect loaders to iter()
        iterable_train_loader = iter(train_loader)
        iterable_test_loader = iter(test_loader)


        # learing rate
        lr = config['lr']


        print('Training Start.........')



        for iteration in range(start_iteration, config['niter'] + 1):

            #=============== TRAIN ===================
            # ------ [ train batch loader ] ---------
            try:
                train_img_names, gt_images, gt_targets, orig_images = iterable_train_loader.next()
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                train_img_names, gt_images, gt_targets, orig_images = iterable_train_loader.next()

            # ------ [ train batch  ] ---------
            if cuda:
                orig_images = orig_images.cuda()
                gt_images = gt_images.cuda()



            # Forward
            loss, predict = trainer(orig_images, gt_images)

            if not loss.dim() == 0:
                loss = torch.mean(loss)

            # Backward (update optimizer)
            trainer_module.optimizerSGD.zero_grad()
            loss.backward()
            trainer_module.optimizerSGD.step()


            # [print loss] (in this, 1 print for 30 iteration)
            if iteration % 50 == 0:
                print("Epoch [%d/%d] Loss: %.10f lr:%.6f" % (iteration ,config['niter'], loss.data, lr))


            #=============== TEST ===================
            if iteration % config['viz_iter'] == 0:
                try:
                    test_img_names, test_orig_images = iterable_test_loader.next()
                except StopIteration:
                    iterable_test_loader = iter(test_loader)
                    test_img_names, test_orig_images = iterable_test_loader.next()

                if cuda:
                    test_orig_images = test_orig_images.cuda()

                # <predict test set>
                test_predict = trainer.module.netParser(test_orig_images)

                for test_idx in range(config['test_batch']):
                    pred_out = torch.argmax(test_predict[test_idx], dim=0)
                    test_sam = pred_out.cpu().numpy()

                    if config['save_result_as_colored']:
                        decoded = decode_segmap(test_sam)
                        misc.imsave(os.path.join(config['output_test_dir'], test_img_names[test_idx].split('.')[0] + '.png'), decoded)
                    else:
                        cv2.imwrite(os.path.join(config['output_test_dir'], test_img_names[test_idx].split('.')[0] + '.png'), test_sam)



            # <learning rate up>
            if iteration % 50000 == 0:
                lr = lr * config['lr_decay']
                for param_group in trainer_module.optimizerSGD.param_groups:
                    param_group['lr'] = lr

            # save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(checkpoint_path, iteration)


    except Exception as e:  # for unexpected error logging
        logger.error(f"{e}")
        raise e


if __name__ == '__main__':
    main()
