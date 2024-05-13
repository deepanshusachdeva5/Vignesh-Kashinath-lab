# %%
import os
import cv2
import glob
import torch
import random
import numpy as np
import collections
from PIL import Image
import torch.nn as nn
from skimage import exposure
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import torch.optim as optim
import mrcfile 

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, TensorDataset

from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import torch
import torch.nn.utils
from torchmetrics import F1Score
import pandas as pd




import os
import cv2
import glob
import cc3d 
import csv
import random
import rdp 
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
from tifffile import imread, imwrite


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset


from skimage import exposure,morphology
from sklearn.model_selection import train_test_split

import qlty
from qlty import qlty2D


import albumentations as A
# import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
# import albumentations.augmentations.transforms as T

# %%
basedir = "/data/Chromatin/MultiScale/Chromatin"
images_dir = os.path.join(basedir, "denoised/denoised")
masks_dir = os.path.join(basedir, "segmented_binary/segmented_binary")

train_imgs=[]
for f in sorted(os.listdir(images_dir)):
    img = cv2.imread(os.path.join(images_dir,f), cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (256,256))
    train_imgs.append(img)
train_imgs = np.array(train_imgs)

#print(len(train_imgs))
train_masks=[]
# mask_shape 
for f in sorted(os.listdir(masks_dir)):
    # print(f)
    mask = cv2.imread(os.path.join(masks_dir,f), cv2.IMREAD_GRAYSCALE)
    mask[mask<20]=0
    mask[mask>=20]=1
    mask_shape = mask.shape
    train_masks.append(mask)
#print(len(train_masks))
zero_mask = np.zeros(mask_shape)

# =========================================
#for i in range(4):
#    train_masks.insert(0, zero_mask)
#for i in range(4):
#    train_masks.append(zero_mask)

# =============================================


# for m in train_masks:
#     print(m.shape)
# train_masks = np.array(train_masks)
# # print(np.histogram(train_masks, bins=256))
# masks = np.zeros(train_imgs.shape)

# print(train_masks.shape, masks.shape)
# masks[0:train_masks.shape[0],0:train_masks.shape[1],0:train_masks.shape[2]] = train_masks
# train_masks = masks


# train_masks = train_masks/64
# train_masks = train_masks.astype('uint8')

# print(np.unique(train_masks))




train_imgs = np.array(train_imgs)
train_masks = np.array(train_masks)

labels = len(np.unique(train_masks))
print(np.histogram(train_masks, bins=labels))
# sz = train_imgs.shape
# sz1=list(sz)
# sz1[0] = sz[0]
# sz1[1]= max(512,sz[1])
# sz1[2]= max(512,sz[2])
# print(sz)
# imgs = np.zeros(tuple(sz1), dtype=train_imgs.dtype)
# masks = np.zeros(tuple(sz1), dtype=train_masks.dtype)
# imgs[:sz[0],:sz[1],:sz[2]] = train_imgs
# masks[:sz[0],:sz[1],:sz[2]] = train_masks
# train_imgs = imgs
# train_masks = masks
# %%
train_imgs = train_imgs.astype('float32')
train_masks = train_masks.astype('uint8')
print(train_imgs.shape, train_imgs.dtype)
print(train_masks.shape, train_masks.dtype)


# %%
train_imgs = np.expand_dims(train_imgs, axis=1)
train_masks = np.expand_dims(train_masks, axis=1)


# %%

def shuffle_training(imgs, masks, seed=123):
    x = np.arange(imgs.shape[0])
    random.seed(seed)
    random.shuffle(x)
    #print(x)
    return imgs[x,:], masks[x,:]
np.random.seed()

train_imgs, train_masks = shuffle_training(train_imgs, train_masks, seed=None)
print(train_imgs.shape, train_imgs.dtype)
print(train_masks.shape, train_masks.dtype)

# %%


# %%


# %%

import qlty
from qlty import qlty2D

quilt = qlty2D.NCYXQuilt(X=train_imgs.shape[3],
                        Y=train_imgs.shape[2],
                        window=(256,256),
                        step=(64,64),
                        border=(10,10),
                        border_weight=0)

labeled_imgs = torch.Tensor(train_imgs)
labeled_masks = torch.Tensor(train_masks)
labeled_imgs, labeled_masks = quilt.unstitch_data_pair(labeled_imgs,labeled_masks)

print("x shape: ",train_imgs.shape)
print("y shape: ",train_masks.shape)
print("x_bits shape:", labeled_imgs.shape)
print("y_bits shape:", labeled_masks.shape)

# %%

dicedImgs,dicedMasks = [],[]
for i in range(len(labeled_imgs)):
    # comment this to include all slices even the non annotated slices. 
    if np.unique(labeled_masks[i][0]).shape[0] > 0:
        # bilateral filter
        bilateral = cv2.bilateralFilter(labeled_imgs[i][0].numpy(),5,50,10)
        # clahe equalization 
        clahe = cv2.createCLAHE(clipLimit=3)
        bilateral= bilateral.astype(np.uint16)
        final = clahe.apply(bilateral)
        # Equalize histogram 
        x = exposure.equalize_hist(final)
        dicedImgs.append(x.astype(np.float32))
        dicedMasks.append(labeled_masks[i][0].numpy())

# %%
# %%
train_imgs,train_masks = np.array(dicedImgs),np.array(dicedMasks)
imwrite(os.path.join(basedir, "preprocessed.tif"), train_imgs)
train_imgs,train_masks = np.expand_dims(train_imgs, axis=1),np.expand_dims(train_masks, axis=1)

# train_imgs = labeled_imgs.numpy()
# train_masks = labeled_masks.numpy()
# train_imgs = np.squeeze(train_imgs, axis=1)
# train_masks = np.squeeze(train_masks, axis=1)

# %%
print(train_imgs.shape, train_masks.shape)
# %%
labeled_imgs, labeled_masks = shuffle_training(train_imgs, train_masks)

# %%
def make_loaders(train_data, val_data, test_data, 
                batch_size_train, batch_size_val, batch_size_test):
    
    # can adjust the batch size depending on available memory
    train_loader_params = {'batch_size': batch_size_train,
                     'shuffle': True,
                     'num_workers': num_workers,
                     'pin_memory':True,
                     'drop_last': False}

    val_loader_params = {'batch_size': batch_size_val,
                     'shuffle': False,
                     'num_workers': num_workers,
                     'pin_memory':True,
                     'drop_last': False}

    test_loader_params = {'batch_size': batch_size_test,
                     'shuffle': False,
                     'num_workers': num_workers,
                     'pin_memory':True,
                     'drop_last': False}

    train_loader = DataLoader(train_data, **train_loader_params)
    val_loader = DataLoader(val_data, **val_loader_params)
    test_loader = DataLoader(test_data, **test_loader_params)
    
    return train_loader, val_loader, test_loader

labeled_imgs, labeled_masks = shuffle_training(labeled_imgs, labeled_masks)
"""
# %%
# labeled_imgs, labeled_masks = shuffle_training(labeled_imgs, labeled_masks)
labeled_imgs = torch.Tensor(labeled_imgs)
labeled_masks = torch.Tensor(labeled_masks)
rotated_imgs1 = torch.rot90(labeled_imgs, 1, [2, 3])
rotated_masks1 = torch.rot90(labeled_masks, 1, [2, 3])

rotated_imgs2 = torch.rot90(labeled_imgs, 2, [2, 3])
rotated_masks2 = torch.rot90(labeled_masks, 2, [2, 3])

rotated_imgs3 = torch.rot90(labeled_imgs, 3, [2, 3])
rotated_masks3 = torch.rot90(labeled_masks, 3, [2, 3])

flipped_imgs1 = torch.flip(labeled_imgs, [2])
flipped_masks1 = torch.flip(labeled_masks, [2])

flipped_imgs2 = torch.flip(labeled_imgs, [3])
flipped_masks2 = torch.flip(labeled_masks, [3])

flipped_imgs3 = torch.flip(labeled_imgs, [2,3])
flipped_masks3 = torch.flip(labeled_masks, [2,3])


labeled_imgs = torch.cat((labeled_imgs, rotated_imgs1),0)
labeled_masks = torch.cat((labeled_masks, rotated_masks1),0)

labeled_imgs = torch.cat((labeled_imgs, rotated_imgs2),0)
labeled_masks = torch.cat((labeled_masks, rotated_masks2),0)

labeled_imgs = torch.cat((labeled_imgs, rotated_imgs3),0)
labeled_masks = torch.cat((labeled_masks, rotated_masks3),0)

labeled_imgs = torch.cat((labeled_imgs, flipped_imgs1),0)
labeled_masks = torch.cat((labeled_masks, flipped_masks1),0)

labeled_imgs = torch.cat((labeled_imgs, flipped_imgs2),0)
labeled_masks = torch.cat((labeled_masks, flipped_masks2),0)

labeled_imgs = torch.cat((labeled_imgs, flipped_imgs3),0)
labeled_masks = torch.cat((labeled_masks, flipped_masks3),0)

print('Shape of augmented data:    ', labeled_imgs.shape, labeled_masks.shape)

labeled_imgs, labeled_masks = shuffle_training(labeled_imgs, labeled_masks)


"""
# %%
### Create validation set 

num_val = int(0.2*labeled_imgs.shape[0])
num_total = int(labeled_imgs.shape[0])
num_train = num_total - num_val
print('Number of images for validation: '+ str(num_val))
val_imgs = labeled_imgs[num_train:,:,:]
val_masks = labeled_masks[num_train:,:,:]
train_imgs = labeled_imgs[:num_train,:,:]   # actual training
train_masks = labeled_masks[:num_train,:,:]   # actual training


# %%

print('Size of training data:   ', train_imgs.shape)
print('Size of validation data: ', val_imgs.shape)
print('Size of testing data:    ', val_imgs.shape)

num_labels = np.unique(train_masks)
print('The unique mask labels: ', num_labels)
print("Test labels",np.unique(val_masks))




class CryoETDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images.squeeze(axis=1)
        self.masks = masks.squeeze(axis=1)
        # print(self.images.shape, self.masks.shape)
        # self.images = images.squeeze(axis=1)
        # self.masks = masks.squeeze(axis=1)
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        
        mask = self.masks[idx]
        #print(image.shape, mask.shape)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        #image = np.expand_dims(image, axis=0)
        # mask = np.expand_dims(mask,axis=0)
        # print(image.shape, mask.shape)
        return image, mask


train_transform = A.Compose(
    [
        #A.Resize(256, 256),
        #A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
        #A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #A.HorizontalFlip(p=0.2),
        #A.VerticalFlip(p=0.2),
        #A.RandomScale(scale_limit=0.2, p=0.2),
        A.Resize(512, 512),
        #A.CLAHE(),
        #A.RandomBrightnessContrast(p=0.2),
        #A.RandomGamma(p=0.25),
        #A.RandomToneCurve(p=0.2),   #Requires uint8 image
        #A.AdvancedBlur(p=0.2),
        #A.MultiplicativeNoise(p=0.1),
        #A.ISONoise(p=0.1),
        #A.Emboss(p=1), # Highlights Edges--> Try seperately once
        A.GaussNoise(p=0.2),
        
        ToTensorV2(),
        
    ]
)

# train_data = CryoETDataset(train_imgs, train_masks, transform=train_transform,)

# val_transform = A.Compose(
#     [
#         #A.Resize(256, 256), 
#         #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
#         #A.Emboss(p=1),
#         ToTensorV2()
#     ]
# )

# val_data = CryoETDataset(val_imgs, val_masks, transform=val_transform,)

# test_transform = A.Compose(
#     [
#         #A.Resize(256, 256), 
#         #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
#         #A.Emboss(p=1),
#         ToTensorV2()
#     ]
# )
# test_data = CryoETDataset(val_imgs, val_masks, transform=test_transform,)

train_data = TensorDataset(torch.Tensor(train_imgs), torch.Tensor(train_masks))
val_data = TensorDataset(torch.Tensor(val_imgs), torch.Tensor(val_masks))
test_data = TensorDataset(torch.Tensor(val_imgs), torch.Tensor(val_masks))


# # %%


# %%
# # create data loaders
num_workers = 0   # 1 or 2 work better with CPU, 0 best for GPU

# change batch size based on memory available 
batch_size_train =2
batch_size_val = 1
batch_size_test = 1

train_loader, val_loader, test_loader = make_loaders(train_data,
                                                    val_data,
                                                    test_data,
                                                    batch_size_train, 
                                                    batch_size_val, 
                                                    batch_size_test)


# %%
print((train_masks==0).sum())
print((train_masks==1).sum())
print((train_masks==2).sum())
print((train_masks==3).sum())

# %%
counts=[0,0,0,0]
for i in range(train_masks.shape[0]):
    img = train_masks[i,0]
    counts[0] += (img==0).sum()>0
    counts[1] += (img==1).sum()>0
    counts[2] += (img==2).sum()>0
    counts[3] += (img==3).sum()>0
print(counts)   

# %%

def resulting_conv_size(Hin, dil, pad, stride, ker):
    """
    Computes the resulting size of a tensor dimension given conv input parameters

    Parameters
    ----------
    Hin : input dimension
    dil : dilation
    pad : padding
    stride : stride
    ker : kernsel size

    Returns
    -------
    the size of the resulting tensor

    """
    N0 = (Hin + 2 * pad - dil * (ker - 1) - 1) / stride + 1
    return int(N0)


# %%


def resulting_convT_size(Hin, dil, pad, stride, ker, outp):
    """
    Computes the resulting size of a tensor dimension given convT input parameters

    Parameters
    ----------
    Hin : input dimension
    dil : dilation
    pad : padding
    stride : stride
    ker : kernel size
    outp : the outp parameter

    Returns
    -------
    the size of the resulting tensor
    """
    N0 = (Hin - 1) * stride - 2 * pad + dil * (ker - 1) + outp + 1
    return N0


# %%
def get_outpadding_convT(Nsmall, Nbig, ker, stride, dil, padding):
    """
    Compute the padding and output padding values neccessary for matching
    Nsmall to Nbig dimensionality after an application of nn.ConvTranspose

    :param Nsmall: small array dimensions (start)
    :param Nbig: big array dimension (end)
    :param ker: kernel size
    :param stride: stride
    :param dil: dilation
    :param padding: padding
    :return: the padding and output_padding
    """
    tmp = stride * (Nsmall - 1) - 2 * padding + dil * (ker - 1) + 1
    outp = Nbig - tmp
    # outp = -(Nbig - (Nsmall - 1) * stride - 2*padding + dil * (ker - 1) - 1)
    # outp = int(outp)

    # if tmp % 2 == 0:
    #    outp = 0
    #    padding = int(tmp / 2)
    # else:
    #    outp = 1
    #    padding = int((tmp + 1) / 2)
    #
    # if no_padding == True:
    #    padding = 0

    # assert padding >= 0
    return outp


# %%
def get_outpadding_upsampling(Nsmall, Nbig, factor):
    """
    Computes the extra padding value necessary for matching Nsmall to Nbig
    dimensionality after an application of nn.Upsample

    :param Nsmall: small array dimensions (start)
    :param Nbig: big array dimension (end)
    :param factor: the upsampling sizing factor
    :return: the padding and output_padding
    """
    tmp = Nsmall ** factor
    outp = Nbig - tmp

    return outp


# %%
def conv_padding(dil, kernel):
    """
    Do we need a function for this?
    :param dil: Dilation
    :param kernel: Stride
    :return: needed padding value
    """
    return int(dil * (kernel - 1) / 2)


# %%
def scaling_table(input_size, stride_base, min_power, max_power, kernel):
    """
    A generic scaling table for a variety of possible scale change options.
    :param input_size: input image size
    :param stride_base: the stride_base we want to use
    :param min_power: determines the minimum stride: stride = stride_base**min_power
    :param max_power: determines the maximum stride: stride = stride_base**min_power
    :param kernel: kernel size
    :return: A dict with various settings
    #TODO: DEBUG THIS for stride_base!=2
    """
    # first establish the output sizes with respect to the input these
    # operations are agnostic to dilation sizes as long as padding is chosen
    # properly
    _dil = 1
    _pad = conv_padding(_dil, kernel)

    # get sizes we need to address
    available_sizes = []
    powers = range(min_power, max_power + 1)
    stride_output_padding_dict = {}
    for power in powers:
        # if we scale the image down, we use conv
        if power <= 0:
            stride = stride_base ** (-power)
            out_size = resulting_conv_size(input_size, _dil,
                                           _pad, stride, kernel)
            available_sizes.append(out_size)
            stride_output_padding_dict[power] = {}

        # if we scale up we use conv_transpose
        if power > 0:
            stride = stride_base ** power
            out_size = stride * input_size
            available_sizes.append(out_size)
            stride_output_padding_dict[power] = {}

    # now we need to figure out how to go between different sizes

    for ii in range(len(powers)):
        for jj in range(len(powers)):
            size_A = available_sizes[ii]
            size_B = available_sizes[jj]
            power_A = int(powers[ii])
            power_B = int(powers[jj])
            delta_power = power_B - power_A

            # we have to scale up, so we use conv_transpose
            if delta_power > 0:
                stride = stride_base ** delta_power
                add_pad = size_B - resulting_convT_size(size_A, _dil, _pad,
                                                        stride, kernel, 0)
                stride_output_padding_dict[power_A][power_B] = (stride,
                                                                add_pad)

            else:
                stride = stride_base ** -delta_power
                stride_output_padding_dict[power_A][power_B] = (stride, None)

    return stride_output_padding_dict



# %%
def max_pool_size_result(Nin, kernel, stride, dilation=1, padding=0):
    """
    Determine the spatial dimension size after a max pooling operation

    :param Nin: dimension of 1d array
    :param kernel: kernel size
    :param stride: stride; might need to match kernel size
    :param dilation: dilation factor

    :param padding: padding parameter
    :return: the resulting array length
    """
    Nout = ((Nin + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1
    Nout = int(Nout)
    return Nout



# %%
def unet_sizing_chart(N, depth, stride, maxpool_kernel_size,
                      up_down_padding=0, dilation=1):
    """
    Build a set of dictionaries that are useful to make sure that we can map
    arrays back to the right sizes for each downsampling and upsampling
    operation.

    :param N: dimension of array
    :param depth: the total depth of the unet
    :param stride: the stride - we fix this for a single UNet
    :param maxpool_kernel_size: the max pooling kernel size
    :param up_down_padding: max pooling and convT padding, Default is 0
    :param dilation: the dilation factor. default is 1
    :return: a dictionary with information

    The data associated with key "Sizes" provides images size per depth
    The data associated with key "Pool Setting" provides info needed to
    construct a MaxPool operator The data associated with key "convT
    Setting" provides info need to construct transposed convolutions such
    that the image of a the right size is constructed.

    """
    resulting_sizes = {}
    convT_settings = {}
    pool_settings = {}

    Nin = N
    for ii in range(depth):
        resulting_sizes[ii] = {}
        convT_settings[ii + 1] = {}
        pool_settings[ii] = {}

        Nout = max_pool_size_result(Nin,
                                    stride=stride,
                                    kernel=maxpool_kernel_size,
                                    dilation=dilation,
                                    padding=up_down_padding
                                    )
        # padding=(maxpool_kernel_size - 1) / 2

        pool_settings[ii][ii + 1] = {"padding": up_down_padding,
                                     "kernel": maxpool_kernel_size,
                                     "dilation": dilation,
                                     "stride": stride
                                     }

        resulting_sizes[ii][ii + 1] = (Nin, Nout)

        outp = get_outpadding_convT(Nout, Nin,
                                                  dil=dilation,
                                                  stride=stride,
                                                  ker=maxpool_kernel_size,
                                                  padding=up_down_padding
                                                  )

        Nup = resulting_convT_size(Nout,
                                                 dil=dilation,
                                                 pad=up_down_padding,
                                                 stride=stride,
                                                 ker=maxpool_kernel_size,
                                                 outp=outp
                                                 )

        # assert (Nin == Nup)

        convT_settings[ii + 1][ii] = {"padding": up_down_padding,
                                      "output_padding": outp,
                                      "kernel": maxpool_kernel_size,
                                      "dilation": dilation,
                                      "stride": stride
                                      }

        Nin = Nout

    results = {"Sizes": resulting_sizes,
               "Pool_Settings": pool_settings,
               "convT_settings": convT_settings}
    return results


# %%
def build_up_operator(chart, from_depth, to_depth, in_channels,
                      out_channels, conv_kernel, key="convT_settings"):
    """
    Build an up sampling operator

    :param chart: An array of sizing charts (one for each dimension)
    :param from_depth: The sizing is done at this depth
    :param to_depth: and goes to this depth
    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param conv_kernel: the convolutional kernel we want to use
    :param key: a key we can use - default is fine
    :return: returns an operator
    """
    stride = []
    dilation = []
    kernel = []
    padding = []
    output_padding = []

    for ii in range(len(chart)):
        tmp = chart[ii][key][from_depth][to_depth]
        stride.append(tmp["stride"])
        dilation.append(tmp["dilation"])
        kernel.append(tmp["kernel"])
        padding.append(tmp["padding"])
        output_padding.append(chart[ii][key][from_depth][to_depth]["output_padding"])

    return conv_kernel(in_channels=in_channels,
                       out_channels=out_channels,
                       kernel_size=kernel,
                       stride=stride,
                       padding=padding,
                       output_padding=output_padding)



# %%
def build_down_operator(chart, from_depth, to_depth,
                        maxpool_kernel, key="Pool_Settings"):
    """
    Build a down sampling operator

    :param chart: Array of sizing charts (one for each dimension)
    :param from_depth: we start at this depth
    :param to_depth: and go here
    :param maxpool_kernel: the max pooling kernel we want to use
                                      (MaxPool2D or MaxPool3D)
    :param key: a key we can use - default is fine
    :return: An operator with given specs
    """
    stride = []
    dilation = []
    kernel = []
    padding = []

    for ii in range(len(chart)):
        tmp = chart[ii][key][from_depth][to_depth]
        stride.append(tmp["stride"])
        dilation.append(tmp["dilation"])
        kernel.append(tmp["kernel"])
        padding.append(tmp["padding"])

    return maxpool_kernel(kernel_size=kernel,
                          stride=stride,
                          padding=padding)


# %%


# %%


# %%
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
#count_parameters(tunet3)

# %%


class MultiTUNet(nn.Module):
    """
    This function creates a U-Net model commonly used for image semantic
    segmentation. The model takes in an input image and outputs a segmented
    image, with the number of output classes dictated by the out_channels
    parameter.

    In this dlsia implementation, a number of architecture-governing
    hyperparameters may be tuned by the user, including the network depth,
    convolutional channel growth rate both within & between layers, and the
    normalization & activation operations following each convolution.

    :param image_shape: image shape we use
    :param in_channels: input channels
    :param out_channels: output channels
    :param depth: the total depth
    :param base_channels: the first operator take in_channels->base_channels.
    :param growth_rate: The growth rate of number of channels per depth layer
    :param hidden_rate: How many 'inbetween' channels do we want? This is
                        relative to the feature channels at a given depth
    :param conv_kernel: The convolution kernel we want to us. Conv2D or Conv3D
    :param kernel_down: How do we steps down? MaxPool2D or MaxPool3D
    :param kernel_up: How do we step up? nn.ConvTranspose2d or
                      nn.ConvTranspose3d
    :param normalization: A normalization action
    :param activation: Activation function
    :param conv_kernel_size: The size of the convolutional kernel we use
    :param maxpool_kernel_size: The size of the max pooling kernel we use to
                                step down
    :param stride: The stride we want to use.
    :param dilation: The dilation we want to use.

    """

    def __init__(self,
                 image_shape,
                 in_channels,
                 out_channels,
                 depth,
                 base_channels,
                 growth_rate=2,
                 hidden_rate=1,
                 conv_kernel=nn.Conv2d,
                 kernel_down=nn.MaxPool2d,
                 kernel_up=nn.ConvTranspose2d,
                 normalization=nn.BatchNorm2d,
                 activation=nn.ReLU(),
                 conv_kernel_size=3,
                 maxpool_kernel_size=2,
                 dilation=1
                 ):
        """
        Construct a tuneable UNet

        :param image_shape: image shape we use
        :param in_channels: input channels
        :param out_channels: output channels
        :param depth: the total depth
        :param base_channels: the first operator take in_channels->base_channels.
        :param growth_rate: The growth rate of number of channels per depth layer
        :param hidden_rate: How many 'inbetween' channels do we want? This is
                            relative to the feature channels at a given depth
        :param conv_kernel: instance of PyTorch convolution class. Accepted are
                            nn.Conv1d, nn.Conv2d, and nn.Conv3d.
        :param kernel_down: How do we steps down? MaxPool2D or MaxPool3D
        :param kernel_up: How do we step up? nn.ConvTranspose2d ore
                          nn.ConvTranspose3d
        :param normalization: PyTorch normalization class applied to each
                              layer. Passed as class without parentheses since
                              we need a different instance per layer.
                              ex) normalization=nn.BatchNorm2d
        :param activation: torch.nn class instance or list of torch.nn class
                           instances
        :param conv_kernel_size: The size of the convolutional kernel we use
        :param maxpool_kernel_size: The size of the max pooling/transposed
                                    convolutional kernel we use in
                                    encoder/decoder paths. Default is 2.
        :param stride: The stride we want to use. Controls contraction/growth
                       rates of spatial dimensions (x and y) in encoder/decoder
                       paths. Default is 2.
        :param dilation: The dilation we want to use.
        """
        super().__init__()
        # define the front and back of our network
        self.image_shape = image_shape
        self.in_channels = in_channels
        self.out_channels = out_channels

        # determine the overall architecture
        self.depth = depth
        self.base_channels = base_channels
        self.growth_rate = growth_rate
        self.hidden_rate = hidden_rate

        # These are the convolution / pooling kernels
        self.conv_kernel = conv_kernel
        self.kernel_down = kernel_down
        self.kernel_up = kernel_up

        # These are the convolution / pooling kernel sizes
        self.conv_kernel_size = conv_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size

        # These control the contraction/growth rates of the spatial dimensions
        self.stride = maxpool_kernel_size
        self.dilation = dilation

        # normalization and activation functions
        if normalization is not None:
            self.normalization = normalization
        else:
            self.normalization = None
        if activation is not None:
            self.activation = activation
        else:
            self.activation = None
        self.return_final_layer_ = False

        # we now need to get the sizing charts sorted
        self.sizing_chart = []
        for N in self.image_shape:
            self.sizing_chart.append(unet_sizing_chart(N=N,
                                                       depth=self.depth,
                                                       stride=self.stride,
                                                       maxpool_kernel_size=self.maxpool_kernel_size,
                                                       dilation=self.dilation))

        # setup the layers and partial / outputs
        self.encoder_layer_channels_in = {}
        self.encoder_layer_channels_out = {}
        self.encoder_layer_channels_middle = {}

        self.decoder_layer_channels_in = {}
        self.decoder_layer_channels_out = {}
        self.decoder_layer_channels_middle = {}
        
        self.partials_encoder = {}

        self.encoders = {}
        self.decoders = {}
        for i in range(out_channels):
            self.decoders[i]={}
        self.step_down = {}
        self.step_up={}
        for i in range(out_channels):
            self.step_up[i]={}

        # first pass
        self.encoder_layer_channels_in[0] = self.in_channels
        self.decoder_layer_channels_out[0] = self.base_channels

        for ii in range(self.depth):

            # Match interlayer channels for stepping down
            if ii > 0:
                self.encoder_layer_channels_in[ii] = self.encoder_layer_channels_out[ii - 1]
            else:
                self.encoder_layer_channels_middle[ii] = int(self.base_channels)

            # Set base channels in first layer
            if ii == 0:
                self.encoder_layer_channels_middle[ii] = int(self.base_channels)
            else:
                self.encoder_layer_channels_middle[ii] = int(self.encoder_layer_channels_in[ii] * (self.growth_rate))

            # Apply hidden rate for growth within layers
            self.encoder_layer_channels_out[ii] = int(self.encoder_layer_channels_middle[ii] * self.hidden_rate)

            # Decoder layers match Encoder channels

            # Update decoder layout on 12/18/22. Vanilla version no longer
            # contracts upon middle convolution
            self.decoder_layer_channels_in[ii] = self.encoder_layer_channels_out[ii]
            self.decoder_layer_channels_middle[ii] = self.encoder_layer_channels_out[ii]
            self.decoder_layer_channels_out[ii] = self.encoder_layer_channels_middle[ii]

            # self.decoder_layer_channels_in[ii] = self.encoder_layer_channels_out[ii]
            # self.decoder_layer_channels_middle[ii] = self.encoder_layer_channels_middle[ii]
            # self.decoder_layer_channels_out[ii] = self.encoder_layer_channels_in[ii]

            self.partials_encoder[ii] = None

        # Correct final decoder layer
        self.decoder_layer_channels_out[0] = self.encoder_layer_channels_middle[0]

        # Correct first decoder layer
        self.decoder_layer_channels_in[depth - 2] = self.encoder_layer_channels_in[depth - 1]

        # Second pass, add in the skip connections
        for ii in range(depth - 1):
            self.decoder_layer_channels_in[ii] += self.encoder_layer_channels_out[ii]

        for ii in range(depth):

            if ii < (depth - 1):

                # Build encoder/decoder layers
                self.encoders[ii] = "Encode_%i" % ii
                tmp = self.build_unet_layer(self.encoder_layer_channels_in[ii],
                                            self.encoder_layer_channels_middle[ii],
                                            self.encoder_layer_channels_out[ii])
                self.add_module(self.encoders[ii], tmp)
                for i in range(out_channels):
                    self.decoders[i][ii] = "Decode_"+str(i)+"_"+str(ii) 
                    if ii == 0:
                        tmp = self.build_output_layer(
                            self.decoder_layer_channels_in[ii],
                            self.decoder_layer_channels_middle[ii],
                            self.decoder_layer_channels_out[ii],
                            #self.out_channels,
                            1)
                        self.add_module(self.decoders[i][ii], tmp)
                    else:
                        tmp = self.build_unet_layer(self.decoder_layer_channels_in[ii],
                                                self.decoder_layer_channels_middle[ii],
                                                self.decoder_layer_channels_out[ii])
                        self.add_module(self.decoders[i][ii], tmp)
            else:
                self.encoders[ii] = "Final_layer_%i" % ii
                tmp = self.build_unet_layer(self.encoder_layer_channels_in[ii],
                                            self.encoder_layer_channels_middle[
                                                ii],
                                            self.encoder_layer_channels_out[
                                                ii])
                self.add_module(self.encoders[ii], tmp)

            # Build stepping operations
            if ii < self.depth - 1:
                # we step down like this
                self.step_down[ii] = "Step Down %i" % ii
                tmp = build_down_operator(chart=self.sizing_chart,
                                          from_depth=ii,
                                          to_depth=ii + 1,
                                          maxpool_kernel=self.kernel_down,
                                          key="Pool_Settings")
                self.add_module(self.step_down[ii], tmp)
            if (ii >= 0) and (ii < depth - 1):
                # we step up like this
                for i in range(out_channels):
                    self.step_up[i][ii] = "Step Up " + str(i) + "_" + str(ii)
                    if ii == (depth - 2):
                        tmp = build_up_operator(chart=self.sizing_chart,
                                                from_depth=ii + 1,
                                                to_depth=ii,
                                                in_channels=self.encoder_layer_channels_out[ii + 1],
                                                out_channels=self.encoder_layer_channels_out[ii],
                                                conv_kernel=self.kernel_up,
                                                key="convT_settings")
                    else:
                        tmp = build_up_operator(chart=self.sizing_chart,
                                                from_depth=ii + 1,
                                                to_depth=ii,
                                                in_channels=self.decoder_layer_channels_out[ii + 1],
                                                out_channels=self.encoder_layer_channels_out[ii],
                                                conv_kernel=self.kernel_up,
                                                key="convT_settings")

                    self.add_module(self.step_up[i][ii], tmp)

    def build_unet_layer(self, in_channels, in_between_channels, out_channels):
        """
        Build a sequence of convolutions with activations functions and
        normalization layers

        :param in_channels: input channels
        :param in_between_channels: the in between channels
        :param out_channels: the output channels
        :return:
        """

        # Preallocate modules to house each skip connection modules
        modules = []

        # Add first convolution
        modules.append(self.conv_kernel(in_channels,
                                        in_between_channels,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(in_between_channels))
        if self.activation is not None:
            modules.append(self.activation)

        # Add second convolution
        modules.append(self.conv_kernel(in_between_channels,
                                        out_channels,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(out_channels))
        if self.activation is not None:
            modules.append(self.activation)
        modules.append(nn.Dropout(p=0.50))
        # Finally, wrap all modules together in nn.Sequential
        operator = nn.Sequential(*modules)

        return operator

    def build_output_layer(self, in_channels,
                           in_between_channels1,
                           in_between_channels2,
                           final_channels):
        """
        For final output layer, builds a sequence of convolutions with
        activations functions and normalization layers

        :param final_channels: The output channels
        :type final_channels: int
        :param in_channels: input channels
        :param in_between_channels1: the in between channels after first convolution
        :param in_between_channels2: the in between channels after second convolution
        "param final_channels: number of channels the network outputs
        :return:
        """

        # Preallocate modules to house each skip connection modules
        modules = []

        # Add first convolution
        modules.append(self.conv_kernel(in_channels,
                                        in_between_channels1,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(in_between_channels1))
        if self.activation is not None:
            modules.append(self.activation)

        # Add second convolution
        modules.append(self.conv_kernel(in_between_channels1,
                                        in_between_channels2,
                                        kernel_size=self.conv_kernel_size,
                                        padding=int((self.conv_kernel_size - 1) / 2)
                                        )
                       )

        # Append normalization/activation bundle, if applicable
        if self.normalization is not None:
            modules.append(self.normalization(in_between_channels2))
        if self.activation is not None:
            modules.append(self.activation)
        #print("Check", in_between_channels1, in_between_channels2, final_channels)
        # Append final output convolution
        modules.append(self.conv_kernel(in_between_channels2,
                                        final_channels,
                                        kernel_size=1
                                        )
                       )

        # Finally, wrap all modules together in nn.Sequential
        operator = nn.Sequential(*modules)

        return operator

    def forward(self, x):
        """
        Default forward operator.

        :param x: input tensor.
        :return: output of neural network
        """
        #print("Start", x.shape)
        # first pass through the encoder
        for ii in range(self.depth - 1):
            # channel magic
            x_out = self._modules[self.encoders[ii]](x)
            #print(" Encoder ", ii, x_out.shape)
            # store this for decoder side processing
            self.partials_encoder[ii] = x_out

            # step down
            if ii < self.depth-2:
                x = self._modules[self.step_down[ii]](x_out)
            else:
                x_step_down = self._modules[self.step_down[ii]](x_out)
            # done
        xout_tensors = []
        for i in range(self.out_channels):
            # last convolution in bottom, no need to stash results
            x = self._modules[self.encoders[self.depth - 1]](x_step_down)
            #print("Last Encoder ", x.shape)
            for ii in range(self.depth - 2, 0, -1):
                x = self._modules[self.step_up[i][ii]](x)
                x = torch.cat((self.partials_encoder[ii], x), dim=1)
                x = self._modules[self.decoders[i][ii]](x)

            x = self._modules[self.step_up[i][0]](x)
            x = torch.cat((self.partials_encoder[0], x), dim=1)
            #print(x.shape)
            x_out = self._modules[self.decoders[i][0]](x)
            xout_tensors.append(x_out)
            """
            if i==0:
                x_out = self._modules[self.decoders[i][0]](x)
            else:
                x_out_partial = self._modules[self.decoders[i][0]](x)
                print(x_out.shape, x_out_partial.shape)
                xout = torch.cat((x_out, x_out_partial),dim=1)
            """
        x_out = torch.cat(xout_tensors, dim=1)
        #print(x_out.shape)
        return x_out

    def topology_dict(self):
        """
        Get all parameters needed to build this network

        :return: An orderdict with all parameters needed
        :rtype: OrderedDict
        """

        topo_dict = OrderedDict()
        topo_dict["image_shape"] = self.image_shape
        topo_dict["in_channels"] = self.in_channels
        topo_dict["out_channels"] = self.out_channels
        topo_dict["depth"] = self.depth
        topo_dict["base_channels"] = self.base_channels
        topo_dict["growth_rate"] = self.growth_rate
        topo_dict["hidden_rate"] = self.hidden_rate
        topo_dict["conv_kernel"] = self.conv_kernel
        topo_dict["kernel_down"] = self.kernel_down
        topo_dict["kernel_up"] = self.kernel_up
        topo_dict["normalization"] = self.normalization
        topo_dict["activation"] = self.activation
        topo_dict["conv_kernel_size"] = self.conv_kernel_size
        topo_dict["maxpool_kernel_size"] = self.maxpool_kernel_size
        topo_dict["dilation"] = self.dilation
        return topo_dict

    def save_network_parameters(self, name=None):
        """
        Save the network parameters
        :param name: The filename
        :type name: str
        :return: None
        :rtype: None
        """
        network_dict = OrderedDict()
        network_dict["topo_dict"] = self.topology_dict()
        network_dict["state_dict"] = self.state_dict()
        if name is None:
            return network_dict
        torch.save(network_dict, name)

# %%

depth = 5
base_channels = 32
growth_rate = 2
hidden_rate = 1
in_channels = 1
out_channels = len(num_labels)
num_layers = 40             
layer_width = 1 
max_dilation = 15 
normalization = nn.BatchNorm2d

tunet3 = MultiTUNet(image_shape=(train_imgs.shape[2:4]),
            in_channels=in_channels,
            out_channels=out_channels,
            depth=depth,
            kernel_down=nn.MaxPool2d,
            base_channels=base_channels,
            normalization = nn.BatchNorm2d,
            growth_rate=growth_rate,
            hidden_rate=hidden_rate
            )
# print('Number of parameters: ', helpers.count_parameters(tunet3))

# %%
#count_parameters(tunet3)

# %%
# device = helpers.get_device()
device = "cuda:1"
epochs = 60   # Set number of epochs

# =================================================
# class_weights = torch.FloatTensor([1,2,2,3]).to(device)
#criterion = nn.CrossEntropyLoss(weight=class_weights)   # For segmenting >2 classes
criterion = nn.CrossEntropyLoss()   # For segmenting >2 classes
LEARNING_RATE = 1e-3
print(LEARNING_RATE)
# optimizer_msd = optim.Adam(msd_net.parameters(), lr=LEARNING_RATE)
optimizer_tunet3 = optim.Adam(tunet3.parameters(), lr=LEARNING_RATE)

print('Device we will compute on: ', device)   # cuda:0 for GPU. Else, CPU

# %%
experiments = os.path.join(basedir, "Experiments")
if not os.path.exists(experiments):
    os.makedirs(experiments)
newds_path = os.path.join(experiments,'Results_Chromatin_Patch256')
#if os.path.isdir(newds_path) is False: os.mkdir(newds_path)
if not os.path.exists(newds_path):
    os.makedirs(newds_path)
model_msdnet = '/msdnet'
model_tunet3 = '/tunet3'

# %%

tunet3.to(device)   # send network to GPU
torch.cuda.empty_cache()

main_dir = newds_path + model_tunet3
if os.path.isdir(main_dir) is False: os.mkdir(main_dir)

stepsPerEpoch = np.ceil(train_imgs.shape[0]/batch_size_train)
num_steps_down = 2
scheduler = optim.lr_scheduler.StepLR(optimizer_tunet3,
                                 step_size=int(stepsPerEpoch*(epochs/num_steps_down)),
                                 gamma = 0.1,verbose=False)

# %%





def segmentation_metrics(preds, target, missing_label=-1, are_probs=True, num_classes=None):
    """
    Computes a variety of F1 scores.
    See : https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f

    Parameters
    ----------
    preds : Predicted labels
    target : Target labels
    missing_label : missing label
    are_probs: preds are probabilities (pre or post softmax)
    num_classes: if preds are not probabilities, we need to know the number of classes.

    Returns
    -------
    micro and macro F1 scores
    """

    if are_probs:
        num_classes = preds.shape[1]
        tmp = torch.argmax(preds, dim=1)
    else:
        tmp = preds

    assert num_classes is not None
    
    F1_eval_macro = F1Score(num_classes=num_classes,
                            average='macro')
    F1_eval_micro = F1Score(num_classes=num_classes,
                            average='micro')    

#    F1_eval_macro = F1Score(task='multiclass',
#                            num_classes=num_classes,
#                            average='macro')
#    F1_eval_micro = F1Score(task='multiclass',
#                            num_classes=num_classes,
#                            average='micro')
    a = tmp.cpu()
    b = target.cpu()

    sel = b == missing_label
    a = a[~sel]
    b = b[~sel]
    tmp_macro = torch.Tensor([0])
    tmp_micro = torch.Tensor([0])
    if len(a.flatten()) > 0:
        tmp_macro = F1_eval_macro(a, b)
        tmp_micro = F1_eval_micro(a, b)

    return tmp_micro, tmp_macro


def train_segmentation(net, trainloader, validationloader, NUM_EPOCHS,
                       criterion, optimizer, device,
                       savepath=None, saveevery=None,
                       scheduler=None, show=0,
                       use_amp=False, clip_value=None):
    """
    Loop through epochs passing images to be segmented on a pixel-by-pixel
    basis.

    :param net: input network
    :param trainloader: data loader with training data
    :param validationloader: data loader with validation data
    :param NUM_EPOCHS: number of epochs
    :param criterion: target function
    :param optimizer: optimization engine
    :param device: the device where we calculate things
    :param savepath: filepath in which we save networks intermittently
    :param saveevery: integer n for saving network every n epochs
    :param scheduler: an optional schedular. can be None
    :param show: print stats every n-th epoch
    :param use_amp: use pytorch automatic mixed precision
    :param clip_value: value for gradient clipping. Can be None.
    :return: A network and run summary stats
    """

    train_loss = []
    F1_train_trace_micro = []
    F1_train_trace_macro = []

    # Skip validation steps if False or None loaded
    if validationloader is False:
        validationloader = None
    if validationloader is not None:
        validation_loss = []
        F1_validation_trace_micro = []
        F1_validation_trace_macro = []

    best_score = 1e10
    best_index = 0
    best_state_dict = None

    if savepath is not None:
        if saveevery is None:
            saveevery = 1

    for epoch in range(NUM_EPOCHS):
        running_train_loss = 0.0
        running_F1_train_micro = 0.0
        running_F1_train_macro = 0.0
        tot_train = 0.0

        if validationloader is not None:
            running_validation_loss = 0.0
            running_F1_validation_micro = 0.0
            running_F1_validation_macro = 0.0
            tot_val = 0.0
        count = 0

        for data in trainloader:
            count += 1
            noisy, target = data  # load noisy and target images
            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            # print(noisy.shape, target.shape)
            noisy = noisy.to(device)
            target = target.to(device)

            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                target = target.type(torch.LongTensor)
                target = target.to(device).squeeze(1)

            if use_amp is False:
                # forward pass, compute loss and accuracy
                output = net(noisy)
                loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = net(noisy)
                    loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(optimizer)
                scaler.update()

            # update the parameters
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            optimizer.step()


            tmp_micro, tmp_macro = segmentation_metrics(output, target)

            running_F1_train_micro += tmp_micro.item()
            running_F1_train_macro += tmp_macro.item()
            running_train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        # compute validation step
        if validationloader is not None:
            with torch.no_grad():
                for x, y in validationloader:
                    x = x.to(device)
                    y = y.to(device)
                    N_val = y.shape[0]
                    tot_val += N_val
                    if criterion.__class__.__name__ == 'CrossEntropyLoss':
                        y = y.type(torch.LongTensor)
                        y = y.to(device).squeeze(1)

                    # forward pass, compute validation loss and accuracy
                    if use_amp is False:
                        yhat = net(x)
                        val_loss = criterion(yhat, y)
                    else:
                        with torch.cuda.amp.autocast():
                            yhat = net(x)
                            val_loss = criterion(yhat, y)

                    tmp_micro, tmp_macro = segmentation_metrics(yhat, y)
                    running_F1_validation_micro += tmp_micro.item()
                    running_F1_validation_macro += tmp_macro.item()

                    # update running validation loss and accuracy
                    running_validation_loss += val_loss.item()

        loss = running_train_loss / len(trainloader)
        F1_micro = running_F1_train_micro / len(trainloader)
        F1_macro = running_F1_train_macro / len(trainloader)
        train_loss.append(loss)
        F1_train_trace_micro.append(F1_micro)
        F1_train_trace_macro.append(F1_macro)

        if validationloader is not None:
            val_loss = running_validation_loss / len(validationloader)
            F1_val_micro = running_F1_validation_micro / len(validationloader)
            F1_val_macro = running_F1_validation_macro / len(validationloader)
            validation_loss.append(val_loss)
            F1_validation_trace_micro.append(F1_val_micro)
            F1_validation_trace_macro.append(F1_val_macro)

        if show != 0:
            learning_rates = []
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, show) == 0:
                if validationloader is not None:
                    print(
                        f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    print(
                        f'   Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}')
                    print(
                        f'   Micro Training F1: {F1_micro:.4f} | Micro Validation F1: {F1_val_micro:.4f}')
                    print(
                        f'   Macro Training F1: {F1_macro:.4f} | Macro Validation F1: {F1_val_macro:.4f}')
                else:
                    print(
                        f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    print(
                        f'   Training Loss: {loss:.4e} | Micro Training F1: {F1_micro:.4f} | Macro Training F1: {F1_macro:.4f}')

        if validationloader is not None:
            if val_loss < best_score:
                best_state_dict = net.state_dict()
                best_index = epoch
                best_score = val_loss
        else:
            if loss < best_score:
                best_state_dict = net.state_dict()
                best_index = epoch
                best_score = loss

            if savepath is not None:
                torch.save(best_state_dict, savepath + '/net_best')
                print('   Best network found and saved')
                print('')

        if savepath is not None:
            if np.mod(epoch + 1, saveevery) == 0:
                torch.save(net.state_dict(), savepath + '/net_checkpoint')
                print('   Network intermittently saved')
                print('')

    if validationloader is None:
        validation_loss = None
        F1_validation_trace_micro = None
        F1_validation_trace_macro = None

    results = {"Training loss": train_loss,
               "Validation loss": validation_loss,
               "F1 training micro": F1_train_trace_micro,
               "F1 training macro": F1_train_trace_macro,
               "F1 validation micro": F1_validation_trace_micro,
               "F1 validation macro": F1_validation_trace_macro,
               "Best model index": best_index}

    net.load_state_dict(best_state_dict)
    return net, results









import time
start = time.time()
tunet3, results = train_segmentation(
    tunet3,train_loader, val_loader, epochs, 
    criterion, optimizer_tunet3, device,saveevery=3,
    #scheduler=scheduler,
    show=1)   # training happens here

print("Training Time ", time.time()-start)

# %%
plt.figure(figsize=(10,4))
plt.rcParams.update({'font.size': 16})
plt.plot(results['F1 training macro'], linewidth=2, label='training')
plt.plot(results['F1 validation macro'], linewidth=2, label='validation')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('TUnet with ReLU and BatchNorm')
plt.legend()
plt.tight_layout()
plt.savefig(main_dir + '/losses')


# %%

# %%
torch.save(tunet3.state_dict(), main_dir + '/net')

# %%
params = {'image_shape': train_imgs.shape[2:4], 'in_channels': in_channels, 'out_channels': out_channels, 'depth': depth, 'base_channels': base_channels, 'growth_rate': growth_rate, 'hidden_rate': hidden_rate},

np.save(main_dir+'/params.npy',params)


# %%





# %%
def create_network(model_type, params):
    # set model parameters and initialize the network
    if model_type == "SMSNet":
        net = SMSNet.random_SMS_network(**params)
        model_params = {
          "in_channels": net.in_channels,
          "out_channels": net.out_channels,
          "in_shape": net.in_shape,
          "out_shape": net.out_shape,
          "scaling_table": net.scaling_table,
          "network_graph": net.network_graph,
          "channel_count": net.channel_count,
          "convolution_kernel_size": net.convolution_kernel_size,
          "first_action": net.first_action,
          "hidden_action": net.hidden_action,
          "last_action":net.last_action,
        }
        return net, model_params
    elif model_type == 'MultiTUNet':
        net = MultiTUNet(**params)
        return net, params
    else:
        return None, None





# %%
def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """
    n = 7

    indices = np.random.randint(len(array1), size=n)
    print('The indices of the images are ', indices)
    images1 = array1[indices, :]
    images2 = array2[indices, :]
    plt.figure(figsize=(50, 20))
    
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1, vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2, vmin=0, vmax=1)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



# %%



# %%
def normalize(img):
    
    #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    mean = [0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value
    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value
    denominator = np.reciprocal(std, dtype=np.float32)
    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


# %%
def segment_imgs(testloader, net):
    """ Modified for input and no ground truth"""
    torch.cuda.empty_cache()
    
    seg_imgs = []
    noisy_imgs = []
    counter = 0
    with torch.no_grad():
        for batch in testloader:
            noisy = batch
            noisy = noisy[0]
            #noisy = normalize(noisy)
            noisy = torch.FloatTensor(noisy)
            noisy = noisy.to(device)
            output = net(noisy)
            output = F.softmax(output, dim=1)
            if counter == 0:
                seg_imgs = output.detach().cpu()
                noisy_imgs = noisy.detach().cpu()
            else:
                seg_imgs = torch.cat((seg_imgs, output.detach().cpu()), 0)
                noisy_imgs = torch.cat((noisy_imgs, noisy.detach().cpu()), 0)
                
            counter+=1
            del output
            del noisy
            torch.cuda.empty_cache()
    return seg_imgs, noisy_imgs


# %%
def save_stack(imgx, imgy, imgz, d_type = None, return_stacks=True):
    imgx_stack = []
    imgy_stack = []
    imgz_stack = []

    for j in tqdm(range(len(imgx))):
        ix = Image.open(imgx[j])
        iy = Image.open(imgy[j])
        iz = Image.open(imgz[j])
        
        ix.load()
        iy.load()
        iz.load()

        if d_type == None:
            ix = np.array(ix)
            iy = np.array(iy)
            iz = np.array(iz)
        else:
            ix = np.array(ix, dtype=d_type)
            iy = np.array(iy, dtype=d_type)
            iz = np.array(iz, dtype=d_type)

        imgx_stack.append(ix)
        imgy_stack.append(iy)
        imgz_stack.append(iz)

    imgx_stack = np.array(imgx_stack)
    imgy_stack = np.array(imgy_stack)
    imgz_stack = np.array(imgz_stack)
        
    if return_stacks == True:
        return imgx_stack, imgy_stack, imgz_stack



# %%
# results_dir = "/data/Chromatin/MultiScale/Paper/Base/TS_0005/Experiments/Results_TS0005_MultiTUNet_NoDropOut_NoWeights_Test5/tunet3"
results_dir = main_dir

# %%
def imageSplit(quilt,test_imgs):
    dicedImgs = []
    labeled_imgs = torch.Tensor(test_imgs)
    #print(labeled_imgs.shape)
    labeled_imgs = quilt.unstitch(labeled_imgs)
    #print("x shape: ",test_imgs.shape)
    #print("x_bits shape:", labeled_imgs.shape)
    
    for i in range(len(labeled_imgs)):
        bilateral = cv2.bilateralFilter(labeled_imgs[i][0].numpy(),5,50,10)
        clahe = cv2.createCLAHE(clipLimit=3)
        bilateral= bilateral.astype(np.uint16)
        final = clahe.apply(bilateral)
        x = exposure.equalize_hist(final)
        dicedImgs.append(x.astype(np.float32))
        #dicedImgs.append(final.astype(np.float32))
    return np.expand_dims(np.array(dicedImgs), axis=1)



# %%
params = np.load(results_dir + '/params.npy', allow_pickle=True)
params = params[0]
print('The following define the network parameters: ', params)


# %%
# model_type = 'TUNet'
#model_type = 'MSDNet'
model_type = 'MultiTUNet'  

net, model_params = create_network(model_type, params)
net.load_state_dict(torch.load(results_dir + '/net'))


# %%
# device = helpers.get_device()
device='cuda:1'
print('Device we compute on: ', device)
# print('Number of parameters: ', helpers.count_parameters(net))
net.to(device)


# %%
images_dir = "/data/Chromatin/MultiScale/Chromatin/denoised/denoised"
out_folder = '/data/Chromatin/MultiScale/Chromatin/outputs_Chromatin_Patch256'
if os.path.isdir(f'{out_folder}') is False: os.mkdir(f'{out_folder}')


# %%
files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
print('Number of files to segment: ', len(files))
files.sort()


# %%
# test_imgs = np.array(imread(images_file)).astype('float32')
test_imgs = []
# sample_img = np.zeros((720,512))
for i,file in enumerate(files):
    if i%4!=0:
        continue
#     #img = Image.open(f'{images_dir}/{file}')
#     #img.load()
    img = cv2.imread(f'{images_dir}/{file}', cv2.IMREAD_GRAYSCALE)
    r,c = img.shape
    # sample_img[:r,:c] = img
    #print(img.shape)
    #img = cv2.pyrDown(img, dstsize=(c//2, r//2))
    #down_img = cv2.pyrDown(img, dstsize=(c//2, r//2))
    #img = cv2.pyrDown(down_img, dstsize=(c//4, r//4))
    #img = np.array(img, dtype='float32')
#     # Uncomment this if images are RGB 
#     #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_imgs.append(img)
test_imgs = np.array(test_imgs)
imwrite("/data/Chromatin/MultiScale/Chromatin/Tau_imgs.tif", test_imgs)
test_imgs = np.expand_dims(np.array(test_imgs), axis=1)


# %%
# print(np.unique(test_imgs))


# %%
quilt = qlty2D.NCYXQuilt(X=test_imgs.shape[3],
                         Y=test_imgs.shape[2],
                         window=(256,256),
                         step=(64,64),
                         border=(10,10),
                         border_weight=0)



# %%
output_mapper = {1:"Tau", 2:"Membrane", 3:"Ribosomes"}
file_batch=1
out_masks = None 


# %%
for k,v in output_mapper.items():
    if os.path.isdir(f'{out_folder}/{v}') is False: os.mkdir(f'{out_folder}/{v}')
    if os.path.isdir(f'{out_folder}/{v}/segments') is False: os.mkdir(f'{out_folder}/{v}/segments')

start = time.time()
for i in tqdm(range(0,test_imgs.shape[0],file_batch)):
    imgs = test_imgs[i:i+file_batch]
    #print(imgs.shape)
    dicedtestImgs = imageSplit(quilt, imgs)
    
    batch_size = file_batch
    num_workers = 0    #increase to 1 or 2 with multiple GPUs
    test_data = TensorDataset(torch.Tensor(dicedtestImgs))
    test_loader_params = {'batch_size': batch_size,
                     'shuffle': False,
                     'num_workers': num_workers,
                     'pin_memory':True,
                     'drop_last': False}
    test_loader = DataLoader(test_data, **test_loader_params)  
    
    output, input_imgs  = segment_imgs(test_loader, net)
    stitched_output = quilt.stitch(torch.tensor(output))
    o = torch.squeeze(stitched_output[0], 1)
    tunet3_output = torch.argmax(o.cpu()[:,:,:,:].data, dim=1)
    
    masks=tunet3_output.numpy()
    imgs= np.squeeze(imgs,1)
    
    out_masks=masks if out_masks is None else np.vstack((out_masks,masks))
    
    for k,v in output_mapper.items():
        idx=(masks==k)
        structures=np.zeros(imgs.shape)
        structures[idx]=imgs[idx]
        out_path = f'{out_folder}/{v}/segments/'
        
        for j in range(structures.shape[0]):
            name = f'{i+j:04}.jpg'
            #print(out_path+name)
            Image.fromarray(structures[j].astype(np.uint8)).save(out_path+name)
        
    del output
    del tunet3_output
    del input_imgs
    torch.cuda.empty_cache()
    
imwrite(out_folder+'/masks.tif', np.array(out_masks, 'uint8'))
print("Testing Time ", time.time()-start)


# # %%
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from tifffile import imread, imwrite
# from sklearn.metrics import f1_score, confusion_matrix, classification_report, jaccard_score


# # %%
# from tqdm import tqdm
# from skimage import exposure,morphology
# def clean_stack(img_stack, size):
#     cleaned = np.copy(img_stack)
#     cleaned_index = (cleaned!=0)
#     for j in tqdm(range(len(cleaned))):
#         img = cleaned_index[j,:] 
#         img = morphology.remove_small_objects(img, size, connectivity=1)
#         target_img = cleaned[j,:,:]
#         cleaned[j,:,:] = np.multiply(target_img, img)
#     return np.array(cleaned)


# # %%
# def read(filename, start, end, dtype, clean=True, size=75):
#     img = imread(filename)
#     img = np.array(img, dtype=dtype)
#     img = img[start:end]
#     if clean:
#         img = clean_stack(img, size)
#     return img


# # %%
# def combined_labels(ribo, mem, start, end):
#     img = np.zeros(ribo.shape, dtype='uint8')
#     img[mem!=0] = 1
#     img[ribo!=0] = 2
#     return img


# # %%
# def metrics(img, gt):
    
    
#     #gt = read(gt_filename, start, end, dtype='uint8')
    
#     img = img.flatten()
#     gt = gt.flatten()
    
#     f1_macro = f1_score(gt, img, average='macro')
#     f1_micro = f1_score(gt, img, average='micro')
#     print("Macro F1: ", f1_macro, "\nMicro F1: ", f1_micro)
    
#     cm = confusion_matrix(gt, img)
#     print("Confusion matrix: \n", cm)
    
#     #iou = jaccard_score(gt,img)
#     #print("IoU: ", iou)
#     report = classification_report(gt, img)
#     print("Classification Report : \n", report)


# # %%
# def read_folder(folder, start, end, clean=True, size=75):
#     files = sorted(os.listdir(folder))
#     images = []
#     for i in range(start,end):
#         f = files[i]
#         if f.endswith(".jpg"):
#             img = cv2.imread(os.path.join(folder,f), cv2.IMREAD_GRAYSCALE)
            
#             img[img<50] = 0
#             img[img>50] = 1
#             img = img.astype('bool')
#             images.append(img)
#         elif f.endswith(".tiff"):
#             img = imread(os.path.join(folder,f))
#             img = np.array(img)
#             img = img.astype('bool')
#             images.append(img)
#     images = np.array(images)
#     if clean:
#         images = clean_stack(images, size)
#     return images


# # %%
# # gt = imread("/data/Chromatin/MultiScale/Paper/Base/ChromatinData/masks.tiff")
# # metrics(out_masks, gt_ribo_TS)


# # %%
# # gt_ribo_TS = imread("/data/Chromatin/MultiScale/Paper/Base/TS_0005/masks.tiff")
# # metrics(out_masks, gt_ribo_TS)


# # %%
# masks = imread(os.path.join(basedir,"labels.tif"))
# gt_masks = []
# pred_masks = []
# for i in range(masks.shape[0]):
#     if(np.sum(masks[i])>0):
#         gt_masks.append(masks[i])
#         pred_masks.append(out_masks[i])
# pred_masks = np.array(pred_masks)
# gt_masks = np.array(gt_masks)
# metrics(pred_masks, gt_masks)

# # %%



