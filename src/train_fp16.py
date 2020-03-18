import os
import math
import csv
import argparse
import platform

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# Apex handles automatic mixed precision training, https://github.com/nvidia/apex
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    apexImportError = False
except ImportError:
    print('ImportWarning: Please install apex from https://www.github.com/nvidia/apex to run with mixed precision training.')
    print('Continuing training with full precision.')
    apexImportError = True

import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.utils import shuffle

import utils.parse_utils as pu
from config_parser import UserConfigParser, remove, write
from models.Encoder import _Encoder
from models.PLSNet_Encoder import _PLSNet
from models.AffineRegression import _AffineRegression
from models.USARNet import USARNet
from losses.ncc_loss import NCC
from utils.utility_functions import progress_printer, count_parameters, printFeatureMaps, plotFeatureMaps
from utils.affine_transform import affine_transform
from utils.HDF5Data import LoadHDF5File
from utils.data import CreateDataset, GetDatasetInformation, generate_trainPatches


class FileHandler():
    r"""Class to handle creating and writing of loss during training
        Args:
            lr (float): learning rate
            bs (int): batch size
            ps (int): patch size
            st (int): stride
            ns (int): total number of sets
            lossfile (string): path to loss file
    """

    def __init__(self, lr, bs, ps, st, ns, device, lossfile):
        super(FileHandler, self).__init__()

        self.lr = lr
        self.bs = bs
        self.ps = ps
        self.st = st
        self.ns = ns
        self.device = device
        self.fieldnames = ['epoch', 'training_loss', 'validation_loss', 'lr=' + str(self.lr), 'batch_size=' + str(self.bs),
                           'patch_size=' + str(self.ps), 'stride=' + str(self.st),
                           'number_of_datasets=' + str(self.ns), 'device=' + str(self.device)]
        self.lossfile = lossfile

    def create(self):
        """Creates the file to store loss in
        """
        with open(self.lossfile, 'w') as writeFieldnames:
            writeHead = csv.DictWriter(writeFieldnames, fieldnames=self.fieldnames)
            writeHead.writeheader()

    def write(self, epoch, epoch_train_loss, epoch_val_loss):
        """Writes relevant data to the loss-file
            Args:
                epoch (int): current epoch
                epoch_train_loss (float): train loss for the current epoch
                epoch_val_loss (float): val loss for the current epoch
        """
        with open(self.lossfile, 'a') as writeLoss:
            loss = csv.writer(writeLoss, delimiter=',')
            loss.writerow([(epoch + 1), epoch_train_loss, epoch_val_loss])


def parse():
    parser = argparse.ArgumentParser(description='Ultrasound Registration network training')
    parser.add_argument('--model-name',
                        type=str, required=True,
                        help='Name the model (without file extension).')
    parser.add_argument('-lr',
                        type=pu.float_type, default=1e-2,
                        help='Learning rate for optimizer')
    parser.add_argument('-e', '--epochs',
                        type=pu.int_type, default=1,
                        help='Number of steps to run training')
    parser.add_argument('-bs', '--batch-size',
                        type=pu.int_type, default=16,
                        help='Batch size to use for training')
    parser.add_argument('-ps', '--patch-size',
                        type=pu.int_type, default=64,
                        help='Patch size to divide the full volume into')
    parser.add_argument('-st', '--stride',
                        type=pu.int_type, default=20,
                        help='Stride for dividing the full volume')
    parser.add_argument('-N', '--num-sets',
                        type=pu.range_limited_int_type_TOT_NUM_SETS, default=25,
                        help='Total number of sets to use for training')
    parser.add_argument('-dr', '--drop-rate',
                        type=pu.float_type, default=0,
                        help='Drop rate to use in affine regression')
    parser.add_argument('-cvd', '--cuda-visible-devices',
                        type=str, default='0,1',
                        help='Comma delimited (no spaces) list containing ' +
                        'all available CUDA devices')
    parser.add_argument('-ur',
                        type=pu.str2bool, default=False,
                        help='Use regularization with the loss function')
    parser.add_argument('-rh', '--register-hook',
                        type=pu.str2bool, default=False,
                        help='Register hook on layers to print feature maps')
    parser.add_argument('-ft', '--filter-type',
                        type=str, default='Bilateral_lookup',
                        choices={"Bilateral_lookup"},
                        help='Filter type to train network with')
    parser.add_argument('--precision',
                        type=str, default='amp',
                        choices={'amp', 'full'},
                        help='Choose precision to do training. (amp - automatic mixed, full - float32')
    args = parser.parse_args()

    return args


def main():
    global args, user_config

    args = parse()
    user_config = UserConfigParser()

    # Enhance performance
    cudnn.benchmark = True

    # Manual seed for reproducibilty
    torch.manual_seed(0)
    np.random.seed(0)

    # Choose GPU device (0 or 1 available)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    validation_set_ratio = 0.2  # Ratio for amount of samples to use for validation
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    uname = platform.uname()
    print('\n')
    print('Initializing training with the following configuration:')
    print('\n')
    print("=" * 40, "System Information", "=" * 40)
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    print("=" * 40, "GPU Information", "=" * 40)
    print(f"CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")
    print(f"Device: {device}")
    print("=" * 40, "Parameters", "=" * 40)
    print(f"Model name: {args.model_name}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of training sets: {args.num_sets}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Using regularization: {args.ur}")
    print(f"Filter type: {args.filter_type}")
    print(f"Training precision: float32") if apexImportError else print(f"Training precision: {args.precision}")
    print('\n')

    # Defining filepaths
    model_name = os.path.join(user_config.PROJECT_ROOT, user_config.PROJECT_NAME, 'output/models/', '{}.pt'.format(args.model_name))
    lossfile = os.path.join(user_config.PROJECT_ROOT, user_config.PROJECT_NAME, 'output/txtfiles/', 'loss_{}.csv'.format(args.model_name))
    data_files = os.path.join(user_config.DATA_ROOT, 'patient_data_proc_{}/'.format(args.filter_type))
    data_information = os.path.join(user_config.DATA_ROOT, 'dataset_information.csv')

    lossStorage = FileHandler(lr=args.lr, bs=args.batch_size, ps=args.patch_size,
                              st=args.stride, ns=args.num_sets, device=device, lossfile=lossfile)
    lossStorage.create()

    fixed_patches, moving_patches = generate_trainPatches(data_information=data_information,
                                                          data_files=data_files,
                                                          filter_type=args.filter_type,
                                                          patch_size=args.patch_size,
                                                          stride=args.stride,
                                                          device=device,
                                                          voxelsize=voxelsize,
                                                          tot_num_sets=args.num_sets
                                                          )

    fixed_training_patches = fixed_patches[0:math.floor(fixed_patches.shape[0] * (1 - validation_set_ratio)), :]
    moving_training_patches = moving_patches[0:math.floor(moving_patches.shape[0] * (1 - validation_set_ratio)), :]

    fixed_validation_patches = fixed_patches[math.floor(fixed_patches.shape[0] * (1 - validation_set_ratio)):fixed_patches.shape[0], :]
    moving_validation_patches = moving_patches[math.floor(moving_patches.shape[0] * (1 - validation_set_ratio)):moving_patches.shape[0], :]

    print('Number of training samples: ', fixed_training_patches.shape[0])
    print('Number of validation samples: ', fixed_validation_patches.shape[0])
    print('\n')

    # Model configuration
    model_config = network_config()
    encoder = _Encoder(**model_config['ENCODER_CONFIG'])
    affineRegression = _AffineRegression(**model_config['AFFINE_CONFIG'])

    model = USARNet(encoder, affineRegression).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Decide to do FP32 training or mixed precision
    if args.precision == 'amp' and not apexImportError:
        model, optimizer = amp.initialize(model, optimizer)
    elif args.precision == 'amp' and apexImportError:
        print('Error: Apex not found, cannot go ahead with mixed precision training. Continuing with full precision.')

    print('Number of network parameters: ', count_parameters(model))

    criterion = NCC(useRegularization=args.ur, device=device).to(device)

    # Creating loss-storage variables
    epoch_train_loss = torch.zeros(args.epochs).to(device)
    epoch_validation_loss = torch.zeros(args.epochs).to(device)

    print('Initializing training')
    print('\n')

    for epoch in range(args.epochs):

        # Weight for regularization of loss function.
        weight = 0.05

        with torch.autograd.set_detect_anomaly(True):  # Set for debugging possible errors

            model.train()
            training_loss = train(fixed_patches=fixed_training_patches,
                                  moving_patches=moving_training_patches,
                                  epoch=epoch,
                                  model=model,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  weight=weight,
                                  device=device)

        with torch.no_grad():
            model.eval()
            validation_loss = validate(fixed_patches=fixed_validation_patches,
                                       moving_patches=moving_validation_patches,
                                       epoch=epoch,
                                       model=model,
                                       criterion=criterion,
                                       weight=weight,
                                       device=device)

        epoch_train_loss[epoch] = torch.mean(training_loss)
        epoch_validation_loss[epoch] = torch.mean(validation_loss)

        # Save model
        model_info = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch
                      }
        torch.save(model_info, model_name)

        print('Epoch: {}/{} \t Training_loss: {} \t Validation loss: {}'
              .format(epoch + 1, args.epochs, epoch_train_loss[epoch], epoch_validation_loss[epoch]))
        print('\n')

        # Write loss to file
        lossStorage.write(epoch, epoch_train_loss[epoch].item(), epoch_validation_loss[epoch].item())

    print(epoch_train_loss, epoch_validation_loss)


def train(fixed_patches, moving_patches, epoch, model, criterion, optimizer, weight, device):
    r"""Training the model
        Args:
            fixed_patches (Tensor): Tensor holding the fixed_patches ([num_patches, 1, patch_size, patch_size, patch_size])
            moving_patches (Tensor): Tensor holding the moving patches ([num_patches, 1, patch_size, patch_size, patch_size])
            epoch (int): current epoch
            model (nn.Module): Network model
            criterion (nn.Module): Loss-function
            optimizer (optim.Optimizer): optimizer in which to optimise the network
            weight (float): float number to weight the regularizer in the loss function
        Returns:
            array of training losses over each batch
    """

    train_set = CreateDataset(fixed_patches, moving_patches)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

    training_loss = torch.zeros(len(train_loader), device=device)

    for batch_idx, (fixed_batch, moving_batch) in enumerate(train_loader):

        fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

        optimizer.zero_grad()

        predicted_theta = model(fixed_batch, moving_batch)
        predicted_deform = affine_transform(moving_batch, predicted_theta)

        if args.register_hook:
            get_hook(model)

        loss = criterion(fixed_batch, predicted_deform, predicted_theta, weight, reduction='mean')
        training_loss[batch_idx] = loss.item()

        if args.precision == 'amp' and not apexImportError:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        printer = progress_printer(batch_idx / len(train_loader))
        print(printer + ' Training epoch {:2}/{} (steps: {})'.format(epoch + 1, args.epochs, len(train_loader)), end='\r', flush=True)

    return training_loss


def validate(fixed_patches, moving_patches, epoch, model, criterion, weight, device):
    """Validating the model using part of the dataset
        Args:
            fixed_patches (Tensor): Tensor holding the fixed_patches ([num_patches, 1, patch_size, patch_size, patch_size])
            moving_patches (Tensor): Tensor holding the moving patches ([num_patches, 1, patch_size, patch_size, patch_size])
            epoch (int): current epoch
            model (nn.Module): Network model
            criterion (nn.Module): Loss-function
            weight (float): float number to weight the regularizer in the loss function
        Returns:
            array of validation losses over each batch
    """

    validation_set = CreateDataset(fixed_patches, moving_patches)
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=True,
                                   num_workers=2, pin_memory=True)

    validation_loss = torch.zeros(len(validation_loader), device=device)

    for batch_idx, (fixed_batch, moving_batch) in enumerate(validation_loader):

        fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

        predicted_theta = model(fixed_batch, moving_batch)
        predicted_deform = affine_transform(moving_batch, predicted_theta)

        loss = criterion(fixed_batch, predicted_deform, predicted_theta, weight, reduction='mean')
        validation_loss[batch_idx] = loss.item()

        printer = progress_printer((batch_idx + 1) / len(validation_loader))
        print(printer + ' Validating epoch {:2}/{} (steps: {})'.format(epoch + 1, args.epochs, len(validation_loader)), end='\r')

    print('\n')

    return validation_loss


def network_config():
    ENCODER_CONFIG = {'encoder_config': user_config.encoder_config,
                      'growth_rate': user_config.growth_rate,
                      'num_init_features': user_config.num_init_features}

    # Calculating spatial resolution of the output of the encoder
    spatial_resolution = (args.patch_size // ((2**len(user_config.encoder_config))))**3
    num_feature_maps = user_config.num_init_features * 2**(len(user_config.encoder_config) - 1) + 1

    INPUT_SHAPE = spatial_resolution * num_feature_maps * 2

    AFFINE_CONFIG = {'num_input_parameters': INPUT_SHAPE,
                     'num_init_parameters': user_config.num_init_parameters,
                     'affine_config': user_config.affine_config,
                     'drop_rate': args.drop_rate}

    kwargs = {'ENCODER_CONFIG': ENCODER_CONFIG,
              'AFFINE_CONFIG': AFFINE_CONFIG}

    # Write current configuration to main_config.ini to use for prediction at a later stage.
    remove()
    write(bs=args.batch_size, ps=args.patch_size, st=args.stride)

    return kwargs


def get_hook(model):
    # Initial convolutions
    model.encoder.conv0.register_forward_hook(printFeatureMaps)
    model.encoder.conv1.register_forward_hook(printFeatureMaps)
    # Following happens inside the first dilated residual block
    model.encoder.drd_module.DRD_BLOCK1.intermediate.ConvLayer1.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK1.intermediate.ConvLayer2.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK1.intermediate.ConvLayer3.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK1.intermediate.ConvLayer4.register_forward_hook(printFeatureMaps)
    # Following happens at the end of each following dilated residual block
    model.encoder.drd_module.DRD_BLOCK1.conv.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK2.conv.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK3.conv.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK4.conv.register_forward_hook(printFeatureMaps)
    model.encoder.drd_module.DRD_BLOCK5.conv.register_forward_hook(printFeatureMaps)


if __name__ == '__main__':

    # Supress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    main()
