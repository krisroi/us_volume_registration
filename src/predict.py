import torch
from torch.utils.data import Dataset, DataLoader
import csv
import math
import os
import argparse
from datetime import datetime
from sys import platform

from config_parser import UserConfigParser

from models.USARNet import USARNet
from models.Encoder import _Encoder
from models.AffineRegression import _AffineRegression

from losses.ncc_loss import NCC
import utils.parse_utils as pu
from utils.affine_transform import affine_transform
from utils.load_hdf5 import LoadHDF5File
from utils.patch_volume import create_patches
from utils.utility_functions import progress_printer, plotPatchwisePrediction
from utils.data import CreatePredictionSet, generate_predictionPatches


def create_net(model_name, device, ENCODER_CONFIG, AFFINE_CONFIG):
    """Load network model and return in .eval() mode
        Args:
            model_name (string): absolute path to model
            device (torch.device): device to load model on
    """
    encoder = _Encoder(**ENCODER_CONFIG)
    affineRegression = _AffineRegression(**AFFINE_CONFIG)

    net = USARNet(encoder, affineRegression).to(device)

    print('Loading weights ...')
    model = torch.load(model_name, map_location=device)
    net.load_state_dict(model['model_state_dict'])

    return net.eval()


def predict(DATA_ROOT, data_files, filter_type, PROJECT_ROOT, patch_size, stride, device,
            voxelsize, model_name, batch_size, ENCODER_CONFIG, AFFINE_CONFIG, plot_patchwise_prediction):
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
    if platform == 'linux' or platform == 'linux2':
        loc_path = '{}procrustes_analysis/loc_prediction.csv'.format(PROJECT_ROOT)
        theta_path = '{}procrustes_analysis/theta_prediction.csv'.format(PROJECT_ROOT)
    else:
        loc_path = '{}loc_prediction.csv'.format(PROJECT_ROOT)
        theta_path = '{}theta_prediction.csv'.format(PROJECT_ROOT)

    with open(loc_path, 'w') as lctn:
        fieldnames = ['x_pos', 'y_pos', 'z_pos']
        field_writer = csv.DictWriter(lctn, fieldnames=fieldnames)
        field_writer.writeheader()

    with open(theta_path, 'w') as tht:
        fieldnames = ['t1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12']
        field_writer = csv.DictWriter(tht, fieldnames=fieldnames)
        field_writer.writeheader()

    net = create_net(model_name, device, ENCODER_CONFIG, AFFINE_CONFIG)
    criterion = NCC(useRegularization=False, device=device)
    prediction_start_time = datetime.now()

    patch_gen = datetime.now()
    fixed_patches, moving_patches, loc = generate_predictionPatches(DATA_ROOT, data_files, filter_type, patch_size, stride, device, voxelsize)
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

        if plot_patchwise_prediction:
            plotPatchwisePrediction(fixed_batch=fixed_batch,
                                    moving_batch=moving_batch,
                                    predicted_theta=predicted_theta,
                                    PROJ_ROOT=PROJECT_ROOT,
                                    PROJ_NAME=PROJECT_NAME
                                    )

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
    parser.add_argument('-ppw', '--plot_patchwise_prediction',
                        type=pu.str2bool, default=False,
                        help='Plot patchwise predicted alignment')
    args = parser.parse_args()

    # GPU configuration
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    #=================VARIABLE PARAMETERS=====================#
    user_config = UserConfigParser()
    batch_size = user_config.batch_size
    patch_size = user_config.patch_size
    stride = user_config.stride
    #==========================================================#
    #====================FIXED PARAMETERS======================#
    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    filter_type = 'Bilateral_lookup'
    #===========================================================#
    #===============NETWORK SUBMODULES CONFIGURATION============#
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
                     'drop_rate': 0}

    kwargs = {'ENCODER_CONFIG': ENCODER_CONFIG,
              'AFFINE_CONFIG': AFFINE_CONFIG,
              'plot_patchwise_prediction': args.plot_patchwise_prediction
              }
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

    model_name = os.path.join(PROJECT_ROOT, PROJECT_NAME, 'output/models/{}'.format(args.model_name))

    data_files = os.path.join(DATA_ROOT, 'patient_data_proc_{}/'.format(filter_type))  # Path to .h5 data
    #===========================================================#

    with torch.no_grad():
        predict(DATA_ROOT, data_files, filter_type, PROJECT_ROOT, patch_size, stride, device,
                voxelsize, model_name, batch_size, **kwargs)
        print('\n')
