import sys
import scipy.misc as m
import torch.utils.data as data
import PIL
import torch
import numpy as np
from os import listdir
from utils.tools import default_loader, is_image_file, normalize
import os

import torchvision.transforms as transforms



# sample -> gt data FIRST!!
class Parse_Dataset(data.Dataset):
    def __init__(self, data_path, gt_path, image_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(Parse_Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
            #self.samples = self._find_samples_in_subfolders(gt_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]
            #self.samples = [x for x in listdir(gt_path) if is_image_file(x)]

        self.data_path = data_path
        self.gt_path = gt_path # because gt has fewer images, we have to fix the number same with gt images
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name
        print(str(len(self.samples)) + "  items found")

        #--------- SEGMENTATION STATS
        self.n_classes = 17



    def __getitem__(self, index):

        # first, load train image
        img_path = os.path.join(self.data_path, self.samples[index])
        orig_img = default_loader(img_path)
        
        # second, get gt image (same name with '.png')
        gt_path = os.path.join(self.gt_path, self.samples[index])
        gt_path = gt_path.replace('jpg', 'png')
        gt_img = default_loader(gt_path, chan='L')


        if self.random_crop:

            imgw, imgh = orig_img.size
            
            # GT IMAGE
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                gt_img = transforms.Resize(min(self.image_shape))(gt_img)
            gt_img = transforms.RandomCrop(self.image_shape)(gt_img)

            # ORIGINAL IMAGE
            imgw, imgh = orig_img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                orig_img = transforms.Resize(min(self.image_shape))(orig_img)
            orig_img = transforms.RandomCrop(self.image_shape)(orig_img)

        else:
            # GT
            gt_img = transforms.Resize(self.image_shape)(gt_img)
            gt_img = transforms.RandomCrop(self.image_shape)(gt_img)
            # ORIG
            orig_img = transforms.Resize(self.image_shape)(orig_img)
            orig_img = transforms.RandomCrop(self.image_shape)(orig_img)

        # img : (0 ~ 15), we have to convert it to 16 channel
        #img = np.array(img, dtype=np.int32)

        gt_img = np.array(gt_img, dtype=np.uint8)

        #print("max : " + str(img.max()) + ", low : " + str(img.min()))
        #img = transforms.ToTensor()(img).int() # turn the image to a tensor
        gt_img = torch.from_numpy(gt_img).long()
        #print(">max : " + str(img.max()) + ", low : " + str(img.min()))
        #\img = normalize(img)


        orig_img = transforms.ToTensor()(orig_img)  # turn the image to a tensor
        orig_img = normalize(orig_img)

        raw_name = self.samples[index].split('.')[0]

        #print(str(img.size()) + " ---- " + str(orig_img.size()))

        # =============================== OUTPUT IMAGE CAUTION SIZE!!!
        target = np.zeros([self.n_classes, 128, 128])
        #target = np.zeros([self.n_classes, 256, 256])
        for c in range(self.n_classes):
            target[c][gt_img == c] = 1

        target = torch.from_numpy(target).float()




        #lbl_img = m.toimage(img, high=img.max(), low=img.min())
        #print(lbl_img.shape)


        # RETURN (NAME), GT, TARGET(ONE-HOT), ORIG
        if self.return_name:
            return raw_name, gt_img, target, orig_img
        else:
            return gt_img, target, orig_img

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        # item = (path, class_to_idx[target])
                        # samples.append(item)
                        samples.append(path)
        return samples

    def __len__(self):
        return len(self.samples)



