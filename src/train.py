import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import os
import math
import csv
import argparse
import matplotlib.pyplot as plt

from sys import platform
from datetime import datetime
from sklearn.utils import shuffle

from config_parser import UserConfigParser, remove, write

# Folder dependent imports
import utils.parse_utils as pu
from models.Encoder import _Encoder
from models.AffineRegression import _AffineRegression
from models.USARNet import USARNet
from losses.ncc_loss import NCC
from utils.utility_functions import progress_printer, count_parameters
from utils.affine_transform import affine_transform
from utils.load_hdf5 import LoadHDF5File
from utils.data import CreateDataset, GetDatasetInformation, generate_trainPatches


def validate(fixed_patches, moving_patches, epoch, epochs, batch_size, net, criterion, device, weight):
    """Validating the model using part of the dataset
        Args:
            fixed_patches (Tensor): Tensor holding the fixed_patches ([num_patches, 1, patch_size, patch_size, patch_size])
            moving_patches (Tensor): Tensor holding the moving patches ([num_patches, 1, patch_size, patch_size, patch_size])
            epoch (int): current epoch
            epochs (int): total number of epochs
            batch_size (int): Desired batch size
            net (nn.Module): Network model
            criterion (nn.Module): Loss-function
            device (torch.device): Device in which to run the validation
        Returns:
            array of validation losses over each batch
    """

    validation_set = CreateDataset(fixed_patches, moving_patches)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    validation_loss = torch.zeros(len(validation_loader), device=device)

    for batch_idx, (fixed_batch, moving_batch) in enumerate(validation_loader):

        fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

        predicted_theta = net(fixed_batch, moving_batch)
        predicted_deform = affine_transform(moving_batch, predicted_theta)

        loss = criterion(fixed_batch, predicted_deform, predicted_theta, weight, reduction='mean')
        validation_loss[batch_idx] = loss.item()

        printer = progress_printer((batch_idx + 1) / len(validation_loader))
        print(printer + ' Validating epoch {:2}/{} (steps: {})'.format(epoch + 1, epochs, len(validation_loader)), end='\r')

    print('\n')

    return validation_loss


def train(fixed_patches, moving_patches, epoch, epochs,
          batch_size, net, criterion, optimizer, device, weight):
    """Training the model
        Args:
            fixed_patches (Tensor): Tensor holding the fixed_patches ([num_patches, 1, patch_size, patch_size, patch_size])
            moving_patches (Tensor): Tensor holding the moving patches ([num_patches, 1, patch_size, patch_size, patch_size])
            epoch (int): current epoch
            epochs (int): total number of epochs
            batch_size (int): Desired batch size
            net (nn.Module): Network model
            criterion (nn.Module): Loss-function
            optimizer (optim.Optimizer): optimizer in which to optimise the network
            device (torch.device): Device in which to run the validation
        Returns:
            array of training losses over each batch
    """

    train_set = CreateDataset(fixed_patches, moving_patches)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    training_loss = torch.zeros(len(train_loader), device=device)  # Holding training loss over all batch_idx for one training set

    for batch_idx, (fixed_batch, moving_batch) in enumerate(train_loader):

        fixed_batch, moving_batch = fixed_batch.to(device), moving_batch.to(device)

        optimizer.zero_grad()

        predicted_theta = net(fixed_batch, moving_batch)
        predicted_deform = affine_transform(moving_batch, predicted_theta)

        loss = criterion(fixed_batch, predicted_deform, predicted_theta, weight, reduction='mean')
        loss.backward()
        training_loss[batch_idx] = loss.item()
        optimizer.step()

        printer = progress_printer(batch_idx / len(train_loader))
        print(printer + ' Training epoch {:2}/{} (steps: {})'.format(epoch + 1, epochs, len(train_loader)), end='\r', flush=True)

    return training_loss


def train_network(lossfile, model_name, fixed_patches, moving_patches, epochs,
                  lr, batch_size, device, validation_set_ratio,
                  ENCODER_CONFIG, AFFINE_CONFIG, useRegularization):

    encoder = _Encoder(**ENCODER_CONFIG)

    affineRegression = _AffineRegression(**AFFINE_CONFIG)

    net = USARNet(encoder, affineRegression).to(device)

    criterion = NCC(useRegularization=useRegularization, device=device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    print('Number of network parameters: ', count_parameters(net))
    #scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    fixed_training_patches = fixed_patches[0:math.floor(fixed_patches.shape[0] * (1 - validation_set_ratio)), :]
    moving_training_patches = moving_patches[0:math.floor(moving_patches.shape[0] * (1 - validation_set_ratio)), :]

    fixed_validation_patches = fixed_patches[math.floor(fixed_patches.shape[0] * (1 - validation_set_ratio)):fixed_patches.shape[0], :]
    moving_validation_patches = moving_patches[math.floor(moving_patches.shape[0] * (1 - validation_set_ratio)):moving_patches.shape[0], :]

    train_range = range(0, math.floor(fixed_patches.shape[0] * (1 - validation_set_ratio)))
    val_range = range(math.floor(fixed_patches.shape[0] * (1 - validation_set_ratio)), fixed_patches.shape[0])

    print('Number of training samples: ', fixed_training_patches.shape[0])
    print('Number of validation samples: ', fixed_validation_patches.shape[0])
    print('\n')

    # Creating loss-storage variables
    epoch_train_loss = torch.zeros(epochs).to(device)
    epoch_validation_loss = torch.zeros(epochs).to(device)

    print('Initializing training')
    print('\n')

    for epoch in range(epochs):

        weight = (40 / (4 + math.exp(epoch / 4)))

        with torch.autograd.set_detect_anomaly(True):  # Set for debugging possible errors
            # Train model
            net.train()
            training_loss = train(fixed_patches=fixed_training_patches,
                                  moving_patches=moving_training_patches,
                                  epoch=epoch,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  net=net,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  device=device,
                                  weight=weight
                                  )

        # Validate model
        with torch.no_grad():
            net.eval()
            validation_loss = validate(fixed_patches=fixed_validation_patches,
                                       moving_patches=moving_validation_patches,
                                       epoch=epoch,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       net=net,
                                       criterion=criterion,
                                       device=device,
                                       weight=weight
                                       )

        # scheduler.step()

        epoch_train_loss[epoch] = torch.mean(training_loss)
        epoch_validation_loss[epoch] = torch.mean(validation_loss)

        model_info = {'model_state_dict': net.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch
                      }
        torch.save(model_info, model_name)

        print('Epoch: {}/{} \t Training_loss: {} \t Validation loss: {}'
              .format(epoch + 1, epochs, epoch_train_loss[epoch], epoch_validation_loss[epoch]))
        print('\n')

        with open(lossfile, mode='a') as loss:
            loss_writer = csv.writer(loss, delimiter=',')
            loss_writer.writerow([(epoch + 1), epoch_train_loss[epoch].item(), epoch_validation_loss[epoch].item()])

    return epoch_train_loss, epoch_validation_loss


if __name__ == '__main__':

    # Supress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    # Manual seed for reproducibility (both CPU and CUDA)
    torch.manual_seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning-rate',
                        type=pu.float_type, default=1e-2,
                        help='Learning rate for optimizer')
    parser.add_argument('-e', '--num-epochs',
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
    parser.add_argument('-ur', '--use-regularization',
                        type=pu.str2bool, default=False,
                        help='Apply regularization to the loss function')
    args = parser.parse_args()

    # Choose GPU device (0 or 1 available)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    #=================VARIABLE PARAMETERS=====================#
    lr = args.learning_rate  # learning rate
    epochs = args.num_epochs  # number of epochs
    tot_num_sets = args.num_sets  # Total number of sets to use for training (25 max, 1 is used for prediction)
    batch_size = args.batch_size
    patch_size = args.patch_size
    stride = args.stride
    #==========================================================#
    #====================FIXED PARAMETERS======================#
    validation_set_ratio = 0.2
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filter_type = 'Bilateral_lookup'
    #===========================================================#

    #===============NETWORK SUBMODULES CONFIGURATION============#
    user_config = UserConfigParser()
    ENCODER_CONFIG = {'encoder_config': user_config.encoder_config,
                      'growth_rate': user_config.growth_rate,
                      'num_init_features': user_config.num_init_features}

    # Calculating spatial resolution of the output of the encoder
    spatial_resolution = (patch_size // ((2**len(user_config.encoder_config))))**3
    num_feature_maps = user_config.num_init_features * 2**(len(user_config.encoder_config) - 1) + 1

    INPUT_SHAPE = spatial_resolution * num_feature_maps * 2

    AFFINE_CONFIG = {'num_input_parameters': INPUT_SHAPE,
                     'num_init_parameters': user_config.num_init_parameters,
                     'affine_config': user_config.affine_config,
                     'drop_rate': args.drop_rate}

    kwargs = {'ENCODER_CONFIG': ENCODER_CONFIG,
              'AFFINE_CONFIG': AFFINE_CONFIG,
              'useRegularization': args.use_regularization}
    #===========================================================#
    #==========REWRITING CONFIGURATION FILE FOR PREDICTION======#
    remove()
    write(bs=batch_size, ps=patch_size, st=stride)
    #===========================================================#
    #==================DEFINING FILES AND PATHS=================#
    if platform == 'linux' or platform == 'linux2':
        PROJECT_ROOT = '/home/krisroi/'
        PROJECT_NAME = 'us_volume_registration'
        DATA_ROOT = '/mnt/EncryptedFastData/krisroi/'
    elif platform == 'darwin':
        PROJECT_ROOT = '/Users/kristofferroise/master_project/'
        PROJECT_NAME = 'us_volume_registration'
        DATA_ROOT = '/Volumes/external/WD_MY_PASSPORT_ARCHIVE/NTNU_project/'

    model_name = os.path.join(PROJECT_ROOT, PROJECT_NAME, 'output/models/', 'model_latest.pt')  # Model path
    lossfile = os.path.join(PROJECT_ROOT, PROJECT_NAME, 'output/txtfiles/', 'loss_latest.csv')  # Loss path

    data_files = os.path.join(DATA_ROOT, 'patient_data_proc_{}/'.format(filter_type))  # Path to .h5 data
    data_information = os.path.join(DATA_ROOT, 'dataset_information.csv')  # Path to information on .h5 data
    #===========================================================#

    #==================INITIALIZE LOSSFILE======================#
    with open(lossfile, 'w') as els:
        fieldnames = ['epoch', 'training_loss', 'validation_loss', 'lr=' + str(lr), 'batch_size=' + str(batch_size),
                      'patch_size=' + str(patch_size), 'stride=' + str(stride),
                      'number_of_datasets=' + str(tot_num_sets), 'device=' + str(device)]
        epoch_writer = csv.DictWriter(els, fieldnames=fieldnames)
        epoch_writer.writeheader()
    #===========================================================#
    import platform
    uname = platform.uname()
    #==================PRINT CONFIGURATION======================#
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
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Number of training sets: {tot_num_sets}")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Using regularization: {args.use_regularization}")
    print('\n')
    #===========================================================#

    fixed_patches, moving_patches = generate_trainPatches(data_information=data_information,
                                                          data_files=data_files,
                                                          filter_type=filter_type,
                                                          patch_size=patch_size,
                                                          stride=stride,
                                                          device=device,
                                                          voxelsize=voxelsize,
                                                          tot_num_sets=tot_num_sets
                                                          )

    training_loss, validation_loss = train_network(lossfile=lossfile,
                                                   model_name=model_name,
                                                   fixed_patches=fixed_patches,
                                                   moving_patches=moving_patches,
                                                   epochs=epochs,
                                                   lr=lr,
                                                   batch_size=batch_size,
                                                   device=device,
                                                   validation_set_ratio=validation_set_ratio,
                                                   **kwargs
                                                   )

    print('End training loss: ', training_loss)
    print('End validation loss: ', validation_loss)
    print('Model name: ', model_name)
