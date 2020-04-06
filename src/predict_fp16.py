import math
import csv
import os
import argparse
import platform

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Apex handles automatic mixed precision training, https://github.com/nvidia/apex
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
    apexImportError = False
except ImportError:
    print('ImportWarning: Please install apex from https://www.github.com/nvidia/apex to run with mixed precision prediction.')
    print('Continuing prediction with full precision.')
    apexImportError = True

from datetime import datetime

import utils.parse_utils as pu
from config_parser import UserConfigParser
from models.Encoder import _Encoder
from models.PLSNet_Encoder import _PLSNet
from models.AffineRegression import _AffineRegression
from models.USARNet import USARNet
from losses.ncc_loss import NCC, normalized_cross_correlation
from utils.affine_transform import affine_transform
from utils.HDF5Data import LoadHDF5File, SaveHDF5File
from utils.utility_functions import progress_printer, plotPatchwisePrediction
from utils.data import CreateDataset, generate_prediction_patches


class FileHandler():
    r"""Class to handle creating and writing of patch-positions and theta predictions
        Args:
            posFile (string): path to patch-position file
            thetaFile (string): path to theta-prediction file
    """

    def __init__(self, posFile, thetaFile):
        super(FileHandler, self).__init__()

        self.posFieldnames = ['x_pos', 'y_pos', 'z_pos']
        self.thetaFieldnames = []
        for i in range(1, 13):
            self.thetaFieldnames.append('t{}'.format(i))
        self.posFile = posFile
        self.thetaFile = thetaFile

    def create(self):
        """Create the files holding the predictions
        """
        with open(self.posFile, 'w') as writePos:
            writeHead = csv.DictWriter(writePos, fieldnames=self.posFieldnames)
            writeHead.writeheader()
        with open(self.thetaFile, 'w') as writeTheta:
            writeHead = csv.DictWriter(writeTheta, fieldnames=self.thetaFieldnames)
            writeHead.writeheader()

    def write(self, loc, theta):
        """Write predictions to corresponding file.
            Args:
                loc (FloatTensor): Tensor holding patch positions
                theta (FloatTensor): Tensor holding predicted thetas.
            Note:
                Both loc and theta are Tensors that need to be converted to numpy arrays.
                In addiditon, if run on GPU needs to be copied to cpu.
                ex: loc.cpu().numpy()
        """
        with open(self.posFile, 'a') as writePos:
            posWriter = csv.writer(writePos, delimiter=',')
            posWriter.writerows(loc)
        with open(self.thetaFile, 'a') as writeTheta:
            thetaWriter = csv.writer(writeTheta, delimiter=',')
            thetaWriter.writerows(theta)


def parse():
    model_names = sorted(name.split('.')[0].strip() for name in os.listdir(os.path.join(user_config.PROJECT_ROOT,
                                                                                        user_config.PROJECT_NAME,
                                                                                        'output/models/')))
    parser = argparse.ArgumentParser(description='Ultrasound Image registration prediction')
    parser.add_argument('-m', '--model-name',
                        choices=model_names, required=True,
                        type=str,
                        help='Specify wanted model name for prediction.')
    parser.add_argument('-cvd', '--cuda-visible-devices',
                        type=str, default='1',
                        help='Number of desired CUDA core to run prediction on')
    parser.add_argument('-ppw', '--plot_patchwise_prediction',
                        type=pu.str2bool, default=False,
                        help='Plot patchwise predicted alignment')
    parser.add_argument('-ft', '--filter-type',
                        type=str, default='Bilateral_lookup',
                        choices={"Bilateral_lookup", "NLMF_lookup"},
                        help='Filter type for prediction')
    parser.add_argument('-pr', '--precision',
                        type=str, default='full',
                        choices={'amp', 'full'},
                        help='Choose precision to do training. (amp - automatic mixed, full - float32')
    parser.add_argument('-sd', '--save-data',
                        type=pu.str2bool, default=False,
                        help='Enables saving of patchwise predictions as HDF5 data')
    args = parser.parse_args()
    return args


def main():
    global args, user_config

    user_config = UserConfigParser()  # Parse main_config.ini
    args = parse()

    # GPU configuration
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    voxelsize = 7.0000003e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    uname = platform.uname()
    print('\n')
    print('Initializing prediction with the following configuration:')
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
    print(f"Batch size: {user_config.batch_size}")
    print(f"Patch size: {user_config.patch_size}")
    print(f"Stride: {user_config.stride}")
    print(f"Filter type: {args.filter_type}")
    print(f"Prediction precision: float32") if apexImportError else print(f"Prediction precision: {args.precision}")
    print('\n')

    model_name = os.path.join(user_config.PROJECT_ROOT, user_config.PROJECT_NAME,
                              'output/models/{}.pt'.format(args.model_name))
    data_files = os.path.join(user_config.DATA_ROOT,
                              'patient_data_proc_{}/'.format(args.filter_type))

    posFile = os.path.join(user_config.PROJECT_ROOT, 'procrustes_analysis', 'loc_predictions',
                           'loc_prediction_{}.csv'.format(args.model_name))
    thetaFile = os.path.join(user_config.PROJECT_ROOT, 'procrustes_analysis', 'theta_predictions',
                             'theta_prediction_{}.csv'.format(args.model_name))

    predictionStorage = FileHandler(posFile, thetaFile)
    predictionStorage.create()

    # Configuration of the model
    model_config = network_config()
    encoder = _Encoder(**model_config['ENCODER_CONFIG'])
    affineRegression = _AffineRegression(**model_config['AFFINE_CONFIG'])
    model = USARNet(encoder, affineRegression)

    # Load model with existing weights
    print('Loading weights ...')
    loadModel = torch.load(model_name, map_location=device)
    model.load_state_dict(loadModel['model_state_dict'])
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Decide on FP32 prediction or mixed precision
    if args.precision == 'amp' and not apexImportError:
        model = amp.initialize(model)
    elif args.precision == 'amp' and apexImportError:
        print('Error: Apex not found, cannot go ahead with mixed precision prediction. Continuing with full precision.')

    criterion = NCC(useRegularization=False, device=device)

    fixed_patches, moving_patches, loc = generate_prediction_patches(DATA_ROOT=user_config.DATA_ROOT,
                                                                     data_files=data_files,
                                                                     filter_type=args.filter_type,
                                                                     patch_size=user_config.patch_size,
                                                                     stride=user_config.stride,
                                                                     device=device)

    print('\n')
    print('Number of prediction samples: {}'.format(fixed_patches.shape[0]))
    print('\n')

    prediction_set = CreateDataset(fixed_patches, moving_patches, loc)
    prediction_loader = DataLoader(prediction_set, batch_size=user_config.batch_size, 
                                   shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    predicted_theta_tmp = torch.zeros([1, user_config.batch_size, 12]).type(dtype).to(device)
    pos_tmp = torch.zeros([1, user_config.batch_size, 3]).type(dtype).to(device)

    sampleNumber = 1  # Hold index for writing correctly to .h5 file
    saveData = SaveHDF5File(user_config.DATA_ROOT)  # Initialize file for saving patches

    print('Predicting')

    # No grads to be computed
    with torch.no_grad():

        # Eval mode for prediction
        model.eval()

        for batch_idx, (fixed_batch, moving_batch, loc) in enumerate(prediction_loader):

            model_rt = datetime.now()
            predicted_theta = model(fixed_batch, moving_batch)
            print('Model runtime: ', datetime.now() - model_rt)

            warped_batch = affine_transform(moving_batch, predicted_theta)

            # Saves the patches as HDF5 data
            if args.save_data:
                saveData.save_hdf5(fixed_batch=fixed_batch,
                                   moving_batch=moving_batch,
                                   warped_batch=warped_batch,
                                   sampleNumber=sampleNumber)

            if args.plot_patchwise_prediction:
                plotPatchwisePrediction(fixed_batch=fixed_batch.cpu(),
                                        moving_batch=moving_batch.cpu(),
                                        predicted_theta=predicted_theta.cpu(),
                                        PROJ_ROOT=user_config.PROJECT_ROOT,
                                        PROJ_NAME=user_config.PROJECT_NAME
                                        )

            predicted_theta = predicted_theta.view(-1, 12)
            predicted_theta_tmp = predicted_theta.type(dtype)
            loc_tmp = loc.type(dtype)

            predictionStorage.write(loc=loc_tmp.cpu().numpy().round(5),
                                    theta=predicted_theta_tmp.cpu().numpy()
                                    )

            sampleNumber += user_config.batch_size
            
            preWarpNcc = normalized_cross_correlation(fixed_batch, moving_batch, reduction=None)
            postWarpNcc = normalized_cross_correlation(fixed_batch, warped_batch, reduction=None)
            
            print_patchloss(preWarpNcc, postWarpNcc)
            

def print_patchloss(preWarpNcc, postWarpNcc):
    
    print('*'*100)
    print('Patch num in batch' + ' | ' + 'NCC before warping' + ' | ' + 'NCC after warping' + ' | ' + 
          'Improvement' + ' | ' + 'Percentwice imp.')
    
    for idx in range(preWarpNcc.shape[0]):
        pre = preWarpNcc[idx,:].item()
        post = postWarpNcc[idx,:].item()
        diff = post - pre
        percent = 100 - ((pre/post)*100)
        print('{:<12}{:>20}{:>20}{:>20}{:>13}%'.format(idx, round(pre, 4), round(post, 4), 
                                                       round(diff, 4), round(percent, 2)))


def network_config():
    """Configuration of the network. Reads data from main_config.ini.
    """
    ENCODER_CONFIG = {'encoder_config': user_config.encoder_config,
                      'growth_rate': user_config.growth_rate,
                      'num_init_features': user_config.num_init_features}

    # Calculating spatial resolution of the output of the encoder
    spatial_resolution = (user_config.patch_size // ((2**len(user_config.encoder_config))))**3
    num_feature_maps = user_config.num_init_features * 2**(len(user_config.encoder_config) - 1) + 1

    INPUT_SHAPE = spatial_resolution * num_feature_maps * 2

    AFFINE_CONFIG = {'num_input_parameters': INPUT_SHAPE,
                     'num_init_parameters': user_config.num_init_parameters,
                     'affine_config': user_config.affine_config,
                     'drop_rate': 0}

    kwargs = {'ENCODER_CONFIG': ENCODER_CONFIG,
              'AFFINE_CONFIG': AFFINE_CONFIG}

    return kwargs


if __name__ == '__main__':

    # Supress warnings
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

    main()
