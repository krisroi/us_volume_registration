import torch
from torch.utils.data import Dataset, DataLoader
import csv
import math
import os
import argparse
from datetime import datetime

from models.USARNet import USARNet
from models.Encoder import _Encoder
from models.AffineRegression import _AffineRegression

import utils.parse_utils as pu
from utils.affine_transform import affine_transform
from utils.load_hdf5 import LoadHDF5File
from utils.patch_volume import create_patches
from utils.ncc_loss import NCC
from utils.utility_functions import progress_printer

class CreatePredictionSet(Dataset):
    """Reads fixed- and moving patches and returns them as a Dataset object for
        use with Pytorch's handy DataLoader.
        Args:
            fixed_patches (Tensor): Tensor containing the fixed patches
            moving_patches (Tensor): Tensor containing the moving patches
        Example:
            dataset = CreateDataset(fixed_patches, moving_patches)
            dataloader = DataLoader(dataset, **kwargs)
    """

    def __init__(self, fixed_patches, moving_patches, patch_location):
        self.fixed_patches = fixed_patches
        self.moving_patches = moving_patches
        self.patch_location = patch_location

    def __len__(self):
        return self.fixed_patches.shape[0]

    def __getitem__(self, idx):
        return self.fixed_patches[idx, :], self.moving_patches[idx, :], self.patch_location[idx, :]


def generate_patches(path_to_h5files, patch_size, stride, device, voxelsize):
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

    fix_set = 'DataStOlavs19to28/p22_3115007/J65BP1R0_proc.h5'
    mov_set = 'DataStOlavs19to28/p22_3115007/J65BP1R2_proc.h5'
    fix_vols = '01'
    mov_vols = '12'

    print('Creating patches ... ')

    vol_data = LoadHDF5File(path_to_h5files, fix_set, mov_set,
                         fix_vols, mov_vols)
    vol_data.normalize()
    vol_data.to(device)

    patched_vol_data, loc = create_patches(vol_data.data, patch_size, stride, device, voxelsize)

    print("Patched vol_data cuda: ", patched_vol_data.is_cuda)

    return patched_vol_data[:, 0, :].unsqueeze(1), patched_vol_data[:, 1, :].unsqueeze(1), loc


def create_net(model_name, device):
    """Load network model and return in .eval() mode
        Args:
            model_name (string): absolute path to model
            device (torch.device): device to load model on
    """
    encoder = _Encoder(**ENCODER_CONFIG)
    
    affineRegression = _AffineRegression(**AFFINE_CONFIG)
    net = USARNet(encoder, affineRegression).to(device)

    print('Loading weights ...')
    model = torch.load(model_name)
    net.load_state_dict(model['model_state_dict'])

    return net.eval()


def predict(path_to_h5files, patch_size, stride, device, voxelsize, model_name, batch_size):
    """Predict global transformation on a prediction set
        Args:
            path_to_h5files (string): absolute path to folder holding .h5 files
            patch_size (int)
            stride (int)
            device (torch.device): device to run prediction on
            voxelsize (float): size of each voxel in a patch
            model_name (string): absolute path to model
            batch_size(int)
    """

    loc_path = '/home/krisroi/procrustes_analysis/loc_prediction.csv'
    theta_path = '/home/krisroi/procrustes_analysis/theta_prediction.csv'

    with open(loc_path, 'w') as lctn:
        fieldnames = ['x_pos', 'y_pos', 'z_pos']
        field_writer = csv.DictWriter(lctn, fieldnames=fieldnames)
        field_writer.writeheader()

    with open(theta_path, 'w') as tht:
        fieldnames = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
        field_writer = csv.DictWriter(tht, fieldnames=fieldnames)
        field_writer.writeheader()

    net = create_net(model_name, device)
    criterion = NCC()
    prediction_start_time = datetime.now()

    patch_gen = datetime.now()
    fixed_patches, moving_patches, loc = generate_patches(path_to_h5files, patch_size, stride, device, voxelsize)
    print('Patch generation runtime: ', datetime.now() - patch_gen)

    print('\n')
    print('Number of prediction samples: {}'.format(fixed_patches.shape[0]))
    print('\n')

    loader_rt = datetime.now()
    prediction_set = CreatePredictionSet(fixed_patches, moving_patches, loc)
    prediction_loader = DataLoader(prediction_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
    print('Loader runtime: ', datetime.now() - loader_rt)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    print('Predicting')

    predicted_theta_tmp = torch.zeros([1, batch_size, 12]).type(dtype).to(device)
    loc_tmp = torch.zeros([1, batch_size, 3]).type(dtype).to(device)

    for batch_idx, (fixed_batch, moving_batch, loc) in enumerate(prediction_loader):

        printer = progress_printer((batch_idx + 1) / len(prediction_loader))
        print(printer, end='\r')

        net_rt = datetime.now()
        predicted_theta = net(fixed_batch, moving_batch)
        print('Net runtime: ', datetime.now() - net_rt)

        predicted_theta = predicted_theta.view(-1, 12)

        predicted_theta_tmp = predicted_theta.type(dtype)
        loc_tmp = loc.type(dtype)

        with open(loc_path, 'a') as lctn:
            lctn_writer = csv.writer(lctn, delimiter=',')
            lctn_writer.writerows((loc_tmp.cpu().numpy().round(5)))

        with open(theta_path, 'a') as tht:
            theta_writer = csv.writer(tht)
            theta_writer.writerows((predicted_theta_tmp.cpu().numpy()))

    print('Prediction runtime: ', datetime.now() - prediction_start_time)


if __name__ == '__main__':
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name',
                        help='Name of the model to use for prediction',
                        required=True)
    parser.add_argument('-cvd', '--cuda-visible-devices',
                        type=str, default='1',
                        help='Number of desired CUDA core to run prediction on')
    parser.add_argument('-bs', '--batch-size',
                        type=pu.int_type, default=16,
                        help='Batch size to use for prediction')
    parser.add_argument('-ps', '--patch-size',
                        type=pu.int_type, default=64,
                        help='Patch size to divide the volume into')
    parser.add_argument('-st', '--stride',
                        type=pu.int_type, default=20,
                        help='Stride for dividing the full volume')
    args = parser.parse_args()
   

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    path_to_h5files = '/mnt/EncryptedFastData/krisroi/patient_data_proc/'
    patch_size = args.patch_size
    stride = args.stride
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voxelsize = 7.0000003e-4
    batch_size = args.batch_size

    with torch.no_grad():
        predict(path_to_h5files, patch_size, stride, device, voxelsize, model_name, batch_size)
        print('\n')
