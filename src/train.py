import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import math
import csv
from datetime import datetime
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# Folder dependent imports
from lib.network import Net
from lib.affine import affine_transform
from lib.HDF5Image import HDF5Image
from lib.patch_volume import create_patches
from lib.ncc_loss import NCC
import lib.utils as ut
from lib.data_info_loader import GetDatasetInformation
from predict import predict


class CreateDataset(Dataset):
    """Reads fixed- and moving patches and returns them as a Dataset object for
        use with Pytorch's handy DataLoader.
        Args:
            fixed_patches (Tensor): Tensor containing the fixed patches
            moving_patches (Tensor): Tensor containing the moving patches
        Example:
            dataset = CreateDataset(fixed_patches, moving_patches)
            dataloader = DataLoader(dataset, **kwargs)
    """

    def __init__(self, fixed_patches, moving_patches):
        self.fixed_patches = fixed_patches
        self.moving_patches = moving_patches

        del fixed_patches, moving_patches

    def __len__(self):
        return self.fixed_patches.shape[0]

    def __getitem__(self, idx):
        return self.fixed_patches[idx, :], self.moving_patches[idx, :]


def progress_printer(percentage):
    """Function returning a progress bar
        Args:
            percentage (float): percentage point
    """
    eq = '=====================>'
    dots = '......................'
    printer = '[{}{}]'.format(eq[len(eq) - math.ceil(percentage * 20):len(eq)], dots[2:len(eq) - math.ceil(percentage * 20)])
    return printer


def weights_init(m):
    """Apply weight and bias initalization
    """
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is None:
            torch.nn.init.zeros_(m.bias)


def generate_patches(path_to_infofile, info_filename, path_to_h5files,
                     patch_size, stride, device, voxelsize, tot_num_sets):
    """Loading all datasets, creates patches and store all patches in a single array.
        Args:
            path_to_file (string): filepath to .txt file containing dataset information
            info_filename (string): filename for the above file
            path_to_h5files (string): path to .h5 files
            patch_size (int): desired patch size
            stride (int): desired stride between patches
            voxelsize (float): not used here, but create_patches has it as input
            tot_num_sets (int): desired number of sets to use in the model
        Returns:
            fixed patches: all fixed patches in the dataset ([num_patches, 1, **patch_size])
            moving patches: all moving patches in the dataset ([num_patches, 1, **patch_size])
    """

    fixed_patches = torch.tensor([]).cpu()
    moving_patches = torch.tensor([]).cpu()

    dataset = GetDatasetInformation(path_to_infofile, info_filename)

    fix_set = dataset.fix_files
    mov_set = dataset.mov_files
    fix_vols = dataset.fix_vols
    mov_vols = dataset.mov_vols

    fix_set, mov_set, fix_vol, mov_vols = shuffle(fix_set, mov_set, fix_vols, mov_vols)

    fix_set = fix_set[0:tot_num_sets]
    mov_set = mov_set[0:tot_num_sets]
    fix_vols = fix_vols[0:tot_num_sets]
    mov_vols = mov_vols[0:tot_num_sets]

    print('Creating patches ... ')

    for set_idx in range(len(fix_set)):

        printer = progress_printer(set_idx / len(fix_set))
        print(printer, end='\r')

        vol_data = HDF5Image(path_to_h5files, fix_set[set_idx], mov_set[set_idx],
                             fix_vols[set_idx], mov_vols[set_idx])
        vol_data.normalize()
        vol_data.to(device)

        patched_vol_data, _ = create_patches(vol_data.data, patch_size, stride, device, voxelsize)
        patched_vol_data = patched_vol_data.cpu()

        fixed_patches = torch.cat((fixed_patches, patched_vol_data[:, 0, :]))
        moving_patches = torch.cat((moving_patches, patched_vol_data[:, 1, :]))

        del patched_vol_data

    print(fixed_patches.shape)

    print('Finished creating patches')

    shuffled_fixed_patches = torch.zeros((fixed_patches.shape[0], patch_size, patch_size, patch_size)).cpu()
    shuffled_moving_patches = torch.zeros((fixed_patches.shape[0], patch_size, patch_size, patch_size)).cpu()

    print(fixed_patches.is_cuda)
    print(moving_patches.is_cuda)

    shuffler = CreateDataset(fixed_patches, moving_patches)
    del fixed_patches, moving_patches
    shuffle_loader = DataLoader(shuffler, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    del shuffler

    print('Shuffling patches ...')

    for batch_idx, (fixed_patches, moving_patches) in enumerate(shuffle_loader):

        printer = progress_printer(batch_idx / len(shuffle_loader))
        print(printer, end='\r')

        shuffled_fixed_patches[batch_idx, :] = fixed_patches
        shuffled_moving_patches[batch_idx, :] = moving_patches

        del fixed_patches, moving_patches

    print('Finished shuffling patches')
    print('\n')

    print("Shuf is cuda: ", shuffled_fixed_patches.is_cuda)

    return shuffled_fixed_patches.unsqueeze(1), shuffled_moving_patches.unsqueeze(1)


def validate(fixed_patches, moving_patches, epoch, epochs, batch_size, net, criterion, device):
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

        predicted_theta = net(moving_batch)
        predicted_deform = affine_transform(moving_batch, predicted_theta)

        loss = criterion(fixed_batch, predicted_deform, reduction='mean')
        validation_loss[batch_idx] = loss.item()

        printer = progress_printer((batch_idx + 1) / len(validation_loader))
        print(printer + ' Validating epoch {:2}/{} (steps: {})'.format(epoch + 1, epochs, len(validation_loader)), end='\r')

    print('\n')

    return validation_loss


def train(fixed_patches, moving_patches, epoch, epochs, batch_size, net, criterion,
          optimizer, device):
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

        predicted_theta = net(moving_batch)
        predicted_deform = affine_transform(moving_batch, predicted_theta)

        loss = criterion(fixed_batch, predicted_deform, reduction='mean')
        loss.backward()
        training_loss[batch_idx] = loss.item()
        optimizer.step()

        printer = progress_printer(batch_idx / len(train_loader))
        print(printer + ' Training epoch {:2}/{} (steps: {})'.format(epoch + 1, epochs, len(train_loader)), end='\r', flush=True)

    return training_loss


def train_network(fixed_patches, moving_patches, epochs, lr, batch_size, path_to_lossfile, device, model_name, validation_set_ratio):

    net = Net().to(device)
    net.apply(weights_init)

    criterion = NCC().to(device)
    optimizer = optim.Adam([
			    {'params': net.stn1.parameters(), 'lr': 1e-3},
			    {'params': net.stn2.parameters(), 'lr': 1e-3},
			    {'params': net.stn3.parameters(), 'lr': 1e-3},
			    {'params': net.sampler1.parameters()},
			    {'params': net.sampler2.parameters()}
			    ], lr=lr)
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

        # Train model
        net.train()
        training_loss = train(fixed_training_patches,
                              moving_training_patches,
                              epoch,
                              epochs,
                              batch_size,
                              net,
                              criterion,
                              optimizer,
                              device,
                              )

        # Validate model
        with torch.no_grad():
            net.eval()
            validation_loss = validate(fixed_validation_patches,
                                       moving_validation_patches,
                                       epoch,
                                       epochs,
                                       batch_size,
                                       net,
                                       criterion,
                                       device
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

        with open(path_to_lossfile, mode='a') as loss:
            loss_writer = csv.writer(loss, delimiter=',')
            loss_writer.writerow([(epoch + 1), epoch_train_loss[epoch].item(), epoch_validation_loss[epoch].item()])

    return epoch_train_loss, epoch_validation_loss


if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)

    # Choose GPU device (0 or 1 available)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    #=======================PARAMETERS==========================#
    lr = 1e-3  # learning rate
    epochs = 150  # number of epochs
    tot_num_sets = 25  # Total number of sets to use for training (25 max, 1 is used for prediction)
    validation_set_ratio = 0.2
    batch_size = 32
    patch_size = 60
    stride = 18
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #===========================================================#

    #=======================SAVING DATA=========================#
    now = datetime.now()
    date = now.strftime('%d%m%Y')
    time = now.strftime('%H%M%S')

    model_name = 'output/models/model_latest_GPURUN.pt'
    #model_name = 'output/models/model_{}_{}.pt'.format(date, time)
    path_to_lossfile = 'output/txtfiles/loss_latest_GPURUN.csv'
    #path_to_lossfile = 'output/txtfiles/avg_loss_{}_epochs_{}_{}.csv'.format(epochs, date, time)

    path_to_h5files = '/mnt/EncryptedFastData/krisroi/patient_data_proc/'
    path_to_infofile = '/mnt/EncryptedFastData/krisroi/'
    info_filename = 'dataset_information.csv'
    #===========================================================#

    #===================INITIALIZE FILES========================#
    with open(path_to_lossfile, 'w') as els:
        fieldnames = ['epoch', 'training_loss', 'validation_loss', 'lr=' + str(lr), 'batch_size=' + str(batch_size),
                      'patch_size=' + str(patch_size), 'stride=' + str(stride),
                      'number_of_datasets=' + str(tot_num_sets), 'device=' + str(device)]
        epoch_writer = csv.DictWriter(els, fieldnames=fieldnames)
        epoch_writer.writeheader()
    #===========================================================#

    start_time = datetime.now()

    generate_patches_start_time = datetime.now()
    fixed_patches, moving_patches = generate_patches(path_to_infofile, info_filename, path_to_h5files,
                                                     patch_size, stride, device, voxelsize, tot_num_sets)
    print('Generate patches runtime: ', datetime.now() - generate_patches_start_time)
    print('\n')

    training_start_time = datetime.now()
    training_loss, validation_loss = train_network(fixed_patches, moving_patches, epochs, lr, batch_size,
                                                   path_to_lossfile, device, model_name, validation_set_ratio)
    print('Training runtime: ', datetime.now() - training_start_time)
    print('\n')

    print('Total runtime: ', datetime.now() - start_time)

    print('End training loss: ', training_loss)
    print('End validation loss: ', validation_loss)
    print('Model name: ', model_name)
