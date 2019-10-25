import math
from urllib.request import urlretrieve

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from scipy.ndimage import zoom 

import torchvision.transforms as transforms
from scipy.misc import imresize


def set_seeds(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.random.manual_seed(1)
    torch.backends.cudnn.deterministic = True

def _worker_init_fn(worker_id):
    np.random.seed(1 + worker_id)

def init_dataset(dataset, train, batch_size, workers):

    # image normalization
    image_normalization_mean = [0.485, 0.456, 0.406]
    image_normalization_std = [0.229, 0.224, 0.225]
    # image_normalization_mean = [0.45, 0.45, 0.45]
    # image_normalization_std = [0.22, 0.22, 0.22]
    normalize = transforms.Normalize(mean=image_normalization_mean,
                                     std=image_normalization_std)
    if train==True:
        dataset.transform = transforms.Compose([
                    # Warp(self.state['image_size']),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        data_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=workers)
    else:
        dataset.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        data_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=workers)
    return dataset, data_loader


def from_tensor_to_ndarray_img(_inputs): # for visualization
    # inp_ = _inputs.clone()
    inp_ = Variable(_inputs)
    inp_ = inp_.data.cpu().numpy()
    # inp_ = inp_ * 0.22
    inp_[0,:,:] = inp_[0,:,:] * 0.229
    inp_[1,:,:] = inp_[1,:,:] * 0.224
    inp_[2,:,:] = inp_[2,:,:] * 0.225

    # inp_ = inp_ + 0.45
    inp_[0,:,:] = inp_[0,:,:] + 0.485
    inp_[1,:,:] = inp_[1,:,:] + 0.456
    inp_[2,:,:] = inp_[2,:,:] + 0.406

    inp_ = np.transpose(inp_, (1, 2, 0))
    return inp_


#--------------------------------- old crop version ------

def four_crops_old(np_images): # size: (batch, channels, size1, size2)
    # print("four crops input shape: ", np_images.shape)
    size1 = np_images.shape[2]
    size2 = np_images.shape[3]
    dim1 = int(size1/2)
    dim2 = int(size2/2)

    # print("crop dimensions: ", (dim1, dim2))
    crop1 = np_images[:, :, 0:dim1, 0:dim2]
    # print("four crops crop1 shape: ", crop1.shape)
    crop2 = np_images[:, :, dim1:2*dim1, 0:dim2]
    # print("four crops crop1 shape: ", crop2.shape)
    crop3 = np_images[:, :, 0:dim1, dim2:2*dim2]
    # print("four crops crop1 shape: ", crop3.shape)
    crop4 = np_images[:, :, dim1:2*dim1, dim2:2*dim2]
    # print("four crops crop1 shape: ", crop4.shape)
    tiles = np.concatenate((crop1, crop2, crop3, crop4), axis=0)
    return tiles

def extract_tiles_old(inputs): # prende in ingresso un tensore e restituisce un tensore eseguendo l'upsample
    inp_ = inputs.data.cpu().numpy()
    tiles_ = four_crops_old(inp_)
    tiles_ = torch.from_numpy(tiles_)
    return tiles_

#-----------------------------------------------------------------


def four_crops(np_images): # size: (batch, channels, size1, size2)
    # print("four crops input shape: ", np_images.shape)
    size1 = np_images.shape[2]
    size2 = np_images.shape[3]
    dim1 = float(size1)/2.0
    dim2 = float(size2)/2.0

    # odd/even patch
    if ((size1 % 2) == 0):
       # print('pari')
       # print('dim1: ', dim1)
       dim1 = int(dim1)
       # print('dim1: ', dim1)
       dim1_1 = dim1
       dim1_2 = dim1
    else:
       # print('dispari')
       # print('dim1: ', dim1)
       dim1 = int(dim1-0.5)
       # print('dim1: ', dim1)
       dim1_1 = dim1+1
       dim1_2 = dim1

    if ((size2 % 2) == 0):
       # print('pari')
       # print('dim2: ', dim2)
       dim2 = int(dim2)
       # print('dim2: ', dim2)
       dim2_1 = dim2
       dim2_2 = dim2
    else:
       # print('dispari')
       # print('dim2: ', dim2)
       dim2 = int(dim2-0.5)
       # print('dim2: ', dim2)
       dim2_1 = dim2+1
       dim2_2 = dim2

    # print("crop dimensions: ", (dim1, dim2))
    crop1 = np_images[:, :, :dim1_1, :dim2_1]
    # print("four crops crop1 shape: ", crop1.shape)
    crop2 = np_images[:, :, dim1_2:, :dim2_1]
    # print("four crops crop1 shape: ", crop2.shape)
    crop3 = np_images[:, :, :dim1_1, dim2_2:]
    # print("four crops crop1 shape: ", crop3.shape)
    crop4 = np_images[:, :, dim1_2:, dim2_2:]
    # print("four crops crop1 shape: ", crop4.shape)
    tiles = np.concatenate((crop1, crop2, crop3, crop4), axis=0)
    return tiles


def extract_tiles(inputs): # prende in ingresso un tensore e restituisce un tensore eseguendo l'upsample
    inp_ = inputs.data.cpu().numpy()
    tiles_ = four_crops(inp_)
    tiles_ = torch.from_numpy(tiles_)
    return tiles_


def count_from_tiles(out_tiles, batch):
    n = int(out_tiles.shape[0]/batch)
    tiles_count = out_tiles[0 : batch]
    # print('(count from tiles) n: ', n)
    # print('(count from tiles) tiles_count shape: ', tiles_count.shape)

    for k in range(1, n):
        tiles_count = tiles_count + out_tiles[k*batch : (k+1)*batch]
        # print('(count from tiles) k: ', k)
        # print('(count from tiles) tiles_count shape: ', tiles_count.shape)

    return tiles_count # out_tiles


def extract_loss_weights(lbs):
    weights = []
    for i in range(0, lbs.shape[0]):
        if lbs[i] < 0.5:
            weights.append(1.0)
        else:
            weights.append(10.0)
            # weights.append(1.0)
    weights = [weights]
    weights = torch.from_numpy(np.array(weights, dtype=np.float32).T)
    weights = Variable(weights.cuda())
    # print ("(extract_loss_weights) labels shape: ", lbs.shape)
    # print ("(extract_loss_weights) loss weights shape: ", weights.shape)
    # print ("(extract_loss_weights) labels: ", lbs)
    # print ("(extract_loss_weights) loss weights: ", weights)
    return weights


def class_from_tiles(labl_tiles, out_tiles_class, weights_, batch):
    n = int(labl_tiles.shape[0]/batch)
    # labl_tiles = F.sigmoid(labl_tiles)

    loss_temp = F.binary_cross_entropy(out_tiles_class, labl_tiles, weight=weights_, reduce=False)

    # print('(class_from_tiles) n: ', n)
    # print('(class_from_tiles) labl_tiles shape: ', labl_tiles.shape)
    # print('(class_from_tiles) out_tiles_class: ', out_tiles_class.shape)
    # print('(class_from_tiles) loss_temp: ', loss_temp.shape)

    class_loss = loss_temp[0:batch]
    for k in range(1, n):
         class_loss = class_loss + loss_temp[k*batch : (k+1)*batch]
    return class_loss


class WarpRect(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size_1 = int(size[0])
        self.size_2 = int(size[1])

        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size_1, self.size_2), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


# -------------------- old util functions ----------------------- #

class Warp(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = int(size)
        self.interpolation = interpolation

    def __call__(self, img):
        # --- added ---
        if self.size <= 0:
            img_new = img
            # print('unresized image!')
        else:
            img_new = img.resize((self.size, self.size), self.interpolation)
        # print('image_size', img_new.size)
        return img_new
        # ------------
        # return img.resize((self.size, self.size), self.interpolation) # original

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)


def download_url(url, destination=None, progress_bar=True):
    """Download a URL to a local file.

    Parameters
    ----------
    url : str
        The URL to download.
    destination : str, None
        The destination of the file. If None is given the file is saved to a temporary directory.
    progress_bar : bool
        Whether to show a command-line progress bar while downloading.

    Returns
    -------
    filename : str
        The location of the downloaded file.

    Notes
    -----
    Progress bar use/example adapted from tqdm documentation: https://github.com/tqdm/tqdm
    """

    def my_hook(t):
        last_b = [0]

        def inner(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            if b > 0:
                t.update((b - last_b[0]) * bsize)
            last_b[0] = b

        return inner

    if progress_bar:
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
            filename, _ = urlretrieve(url, filename=destination, reporthook=my_hook(t))
    else:
        filename, _ = urlretrieve(url, filename=destination)


def conditioned_rmse(prediction, gt):

    print('gt max: ', gt.max().max())
    print('gt shape: ', gt.shape)

    cond_rmse = []
 
    for i in range(0, int(gt.max().max()) + 1):
        band_error = []
        l = 0.0
        for j in range(gt.shape[0]):
            if gt[j] == i:
                band_error.append((gt[j] - prediction[j]))
                l = l + 1.0
        print("gt value: ", i)
        print("num samples", l)

        if l > 0.0:
            band_error = np.array(band_error)
            band_rmse = np.sqrt( ( ( np.square(band_error) ).sum() )/l)
            print("conditioned rmse: ", band_rmse)
        if l == 0.0:
            band_rmse = -1.0
                    
        cond_rmse.append(band_rmse)
        print("---")
    cond_rmse = np.array(cond_rmse)
    return cond_rmse


def interval_rmse(prediction, gt):

    print('gt max: ', gt.max().max())
    print('gt shape: ', gt.shape)

    cond_rmse = []
 
    # ----- GT 0 ---------
    band_error = []
    l = 0.0
    for j in range(gt.shape[0]):
        if gt[j] == 0: # GT equal to 0
            band_error.append((gt[j] - prediction[j]))
            l = l + 1.0
    print("gt value: 0")
    print("num samples", l)

    if l > 0.0:
        band_error = np.array(band_error)
        band_rmse = np.sqrt( ( ( np.square(band_error) ).sum() )/l)
        print("conditioned rmse: ", band_rmse)
    if l == 0.0:
        band_rmse = -1.0
                    
    cond_rmse.append(band_rmse)
    print("---")

    # ----- GT 1-5 -------
    band_error = []
    l = 0.0
    for j in range(gt.shape[0]):
        if (gt[j] >= 1) & (gt[j] <= 5): # GT equal or greater than 0 and equal or minor than 5
            band_error.append((gt[j] - prediction[j]))
            l = l + 1.0

    print("gt values: 1-5")
    print("num samples", l)

    if l > 0.0:
        band_error = np.array(band_error)
        band_rmse = np.sqrt( ( ( np.square(band_error) ).sum() )/l)
        print("conditioned rmse: ", band_rmse)
    if l == 0.0:
        band_rmse = -1.0
                    
    cond_rmse.append(band_rmse)
    print("---")

    # ----- GT 6-10 ------
    band_error = []
    l = 0.0
    for j in range(gt.shape[0]):
        if (gt[j] >= 6) & (gt[j] <= 10): # GT equal or greater than 6 and equal or minor than 10
            band_error.append((gt[j] - prediction[j]))
            l = l + 1.0

    print("gt values: 6-10")
    print("num samples", l)

    if l > 0.0:
        band_error = np.array(band_error)
        band_rmse = np.sqrt( ( ( np.square(band_error) ).sum() )/l)
        print("conditioned rmse: ", band_rmse)
    if l == 0.0:
        band_rmse = -1.0
                    
    cond_rmse.append(band_rmse)
    print("---")

    # ----- GT 11-15 -----
    band_error = []
    l = 0.0
    for j in range(gt.shape[0]):
        if (gt[j] >= 11) & (gt[j] <= 15): # GT equal or greater than 0 and equal or minor than 5
            band_error.append((gt[j] - prediction[j]))
            l = l + 1.0

    print("gt values: 11-15")
    print("num samples", l)

    if l > 0.0:
        band_error = np.array(band_error)
        band_rmse = np.sqrt( ( ( np.square(band_error) ).sum() )/l)
        print("conditioned rmse: ", band_rmse)
    if l == 0.0:
        band_rmse = -1.0
                    
    cond_rmse.append(band_rmse)
    print("---")

    return cond_rmse




class AveragePrecisionMeter(object):
    """
    The APMeter measures the average precision per class.
    The APMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self, difficult_examples=False):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()
        self.difficult_examples = difficult_examples

    def reset(self):
        """Resets the meter with empty member variables"""
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())

    def add(self, output, target):
        """
        Args:
            output (Tensor): NxK tensor that for each of the N examples
                indicates the probability of the example belonging to each of
                the K classes, according to the model. The probabilities should
                sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                         associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                each example (each weight > 0)
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """Returns the model's average precision for each class
        Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        rg = torch.arange(1, self.scores.size(0)).float()

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]

            # compute average precision
            ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):

        # sort examples
        sorted, indices = torch.sort(output, dim=0, descending=True)

        # Computes prec@i
        pos_count = 0.
        total_count = 0.
        precision_at_i = 0.
        for i in indices:
            label = target[i]
            if difficult_examples and label == 0:
                continue
            if label == 1:
                pos_count += 1
            total_count += 1
            if label == 1:
                precision_at_i += pos_count / total_count
        precision_at_i /= pos_count
        return precision_at_i
