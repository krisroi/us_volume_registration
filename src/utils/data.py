import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

from utils.patch_volume import create_patches
from utils.utility_functions import progress_printer
from utils.HDF5Data import LoadHDF5File


class CreateDataset(Dataset):
    """Reads fixed- and moving patches and their respective location and returns a Dataset
        object for use with Pytorch's handy DataLoader.
        Args:
            fixed_patches (Tensor): Tensor containing the fixed patches
            moving_patches (Tensor): Tensor containing the moving patches
            patch_location (Tensor): Tensor containing patch locations. (NoneType for training)
        Example:
            dataset = CreateDataset(fixed_patches, moving_patches)
            dataloader = DataLoader(dataset, **kwargs)
    """

    def __init__(self, fixed_patches, moving_patches, patch_location=None):
        self.fixed_patches = fixed_patches
        self.moving_patches = moving_patches
        self.patch_location = patch_location

        del fixed_patches, moving_patches

    def __len__(self):
        return self.fixed_patches.shape[0]

    def __getitem__(self, idx):
        if self.patch_location is not None:
            return self.fixed_patches[idx, :], self.moving_patches[idx, :], self.patch_location[idx, :]
        else:
            return self.fixed_patches[idx, :], self.moving_patches[idx, :]


class GetDatasetInformation():
    """Reads dataset information from .csv file and returns the information as lists
        Args:
            filename (string): filename of the .csv file containing data
            filter_type (string): pre-processing filter type
    """

    def __init__(self, filename, filter_type, mode):
        super(GetDatasetInformation, self).__init__()

        self.filename = filename
        self.filter_type = filter_type
        self.mode = mode

        self.fix_files, self.mov_files, self.fix_vols, self.mov_vols, self.pid, self.vol_dim = self.load_dataset()

    def get_biggest_dimensions(self):
        """ Get biggest dimension in each direction to use for upsampling
        """
        x_dim = 0
        y_dim = 0
        z_dim = 0
        for dim in self.vol_dim:
            dim = dim.split('x')
            if int(dim[0]) > x_dim:
                x_dim = int(dim[0])
            if int(dim[1]) > y_dim:
                y_dim = int(dim[1])
            if int(dim[2]) > z_dim:
                z_dim = int(dim[2])
                
        return tuple((x_dim, y_dim, z_dim))

    def load_dataset(self):
        """ Reads the dataset information, pulls out the usable datasets
            and returns them together with corresponding volumes.
        """

        data = pd.read_csv(self.filename)
        vol_dim = data.ref_vol_dim.to_list()
        if self.mode == 'training':
            data = data.loc[lambda df: data.usable == 'y', :]  # Extract only usable datasets (y: yes)
        elif self.mode == 'prediction':
            data = data.loc[lambda df: data.usable == 'pred', :]  # Extract only prediction sets
        ref_filename = data.ref_filename
        mov_filename = data.mov_filename
        ref_vol_frame_no = data.ref_vol_frame_no
        mov_vol_frame_no = data.mov_vol_frame_no
        patient_id = data.pid

        # Initializing empty list-holders
        fix_files = []
        mov_files = []
        fix_vols = []
        mov_vols = []
        pid = []

        for _, pat_idx in enumerate((ref_filename.index)):
            fix_files.append('{}_{}.h5'.format(ref_filename[pat_idx], self.filter_type))
            mov_files.append('{}_{}.h5'.format(mov_filename[pat_idx], self.filter_type))
            fix_vols.append('{:02}'.format(ref_vol_frame_no[pat_idx]))
            mov_vols.append('{:02}'.format(mov_vol_frame_no[pat_idx]))
            pid.append('{}'.format(patient_id[pat_idx]))

        return fix_files, mov_files, fix_vols, mov_vols, pid, vol_dim


def shuffle_patches(fixed_patches, moving_patches):
    """Takes two tensors of patches and returns shuffled tensors.
        Args:
            fixed_patches (Tensor): tensor containing fixed patches
            moving_patches (Tensor): tensor containing moving patches
    """

    shuffled_fixed_patches = torch.Tensor(fixed_patches.shape).cpu()
    shuffled_moving_patches = torch.Tensor(moving_patches.shape).cpu()

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

    return shuffled_fixed_patches, shuffled_moving_patches


def generate_train_patches(data_information, data_files, filter_type,
                           patch_size, stride, device, tot_num_sets):
    """Loading all datasets, creates patches and store all patches in a single array.
        Args:
            data_information (string): filename of .csv file containing dataset information
            data_files (string): folder containing .h5 files
            filter_type (string): pre-processing filter type
            patch_size (int): desired patch size
            stride (int): desired stride between patches
            tot_num_sets (int): desired number of sets to use in the model
        Returns:
            fixed patches: all fixed patches in the dataset ([num_patches, 1, **patch_size])
            moving patches: all moving patches in the dataset ([num_patches, 1, **patch_size])
    """

    fixed_patches = torch.tensor([]).cpu()
    moving_patches = torch.tensor([]).cpu()

    dataset = GetDatasetInformation(data_information, filter_type, mode='training')

    fix_set = dataset.fix_files
    mov_set = dataset.mov_files
    fix_vols = dataset.fix_vols
    mov_vols = dataset.mov_vols
    pid = dataset.pid

    fix_set, mov_set, fix_vols, mov_vols, pid = shuffle(fix_set, mov_set, fix_vols, mov_vols, pid)

    fix_set = fix_set[0:tot_num_sets]
    mov_set = mov_set[0:tot_num_sets]
    fix_vols = fix_vols[0:tot_num_sets]
    mov_vols = mov_vols[0:tot_num_sets]
    pid = pid[0:tot_num_sets]

    dims = dataset.get_biggest_dimensions()

    print('Creating patches ... ')
    print('-----------------------------------------------------------------------------------------------------------------')
    print('pid  filename                           vol     patches    Shapes prior to upsampling    Shapes after upsampling')
    print('-----------------------------------------------------------------------------------------------------------------')

    for set_idx in range(len(fix_set)):

        printer = progress_printer(set_idx / len(fix_set))
        print(printer, end='\r')

        hdf_data = LoadHDF5File(data_files, fix_set[set_idx], mov_set[set_idx],
                                fix_vols[set_idx], mov_vols[set_idx], dims)
        
        hdf_data.normalize()
        hdf_data.to(device)

        hdf_data.interpolate_and_concatenate()
        
        '''
        fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(40, 40))

        original_fix = hdf_data.fix_data[0, hdf_data.fix_data.shape[1] // 2].cpu()
        original_mov = hdf_data.mov_data[0, hdf_data.mov_data.shape[1] // 2].cpu()
        interpolated_fix = hdf_data.data[0, hdf_data.data.shape[1] // 2].cpu()
        interpolated_mov = hdf_data.data[1, hdf_data.data.shape[1] // 2].cpu()
        ax[0, 0].imshow(original_fix, origin='left', cmap='gray')
        ax[0, 1].imshow(original_mov, origin='left', cmap='gray')
        ax[1, 0].imshow(interpolated_fix, origin='left', cmap='gray')
        ax[1, 1].imshow(interpolated_mov, origin='left', cmap='gray')
        ax[0, 0].set_xlim([0, hdf_data.fix_data.shape[1]])
        ax[0, 0].set_ylim([hdf_data.fix_data.shape[2], 0])
        ax[0, 1].set_xlim([0, hdf_data.fix_data.shape[1]])
        ax[0, 1].set_ylim([hdf_data.fix_data.shape[2], 0])
        ax[1, 0].set_xlim([0, hdf_data.data.shape[1]])
        ax[1, 0].set_ylim([hdf_data.data.shape[2], 0])
        ax[1, 1].set_xlim([0, hdf_data.data.shape[1]])
        ax[1, 1].set_ylim([hdf_data.data.shape[2], 0])

        plt.show()'''

        patched_vol_data, _ = create_patches(hdf_data.data, patch_size, stride, device)
        patched_vol_data = patched_vol_data.cpu()

        fixed_patches = torch.cat((fixed_patches, patched_vol_data[:, 0, :]))
        moving_patches = torch.cat((moving_patches, patched_vol_data[:, 1, :]))

        print('{:<5}{:>5}{:>10}{:>5}       {}   {}'.format(pid[set_idx], fix_set[set_idx], fix_vols[set_idx],
                                                           patched_vol_data[:, 0, :].shape[0],
                                                           hdf_data.fix_data.shape, hdf_data.data[0, :].shape))
        print('{:<5}{:>5}{:>10}{:>5}       {}   {}'.format(pid[set_idx], mov_set[set_idx], mov_vols[set_idx],
                                                           patched_vol_data[:, 1, :].shape[0],
                                                           hdf_data.mov_data.shape, hdf_data.data[1, :].shape))
        print('-----------------------------------------')

        del patched_vol_data

    print('Finished creating patches')

    shuffled_fixed_patches, shuffled_moving_patches = shuffle_patches(fixed_patches, moving_patches)

    return shuffled_fixed_patches.unsqueeze(1), shuffled_moving_patches.unsqueeze(1)


def generate_prediction_patches(DATA_ROOT, data_files, frame, filter_type, patch_size, stride, device, PSN):
    """Loading all datasets, creates patches and store all patches in a single array.
        Args:
            DATA_ROOT (string): root folder to all data-files
            data_files (string): folder containing the specific .h5 data
            frame (string): filename of information on ED or ES frame
            filter_type (string): filter type used for pre-processing
            patch_size (int): desired patch size
            stride (int): desired stride between patches
            PSN (int): prediction set number to run predictions on
        Returns:
            fixed patches: all fixed patches in the dataset ([num_patches, 1, **patch_size])
            moving patches: all moving patches in the dataset ([num_patches, 1, **patch_size])
            loc: location of each patch
        Note:
            Prediction patches are created and returned on the GPU if GPU is available.
        Issue:
            If prediction fix and moving volumes are of different sizes, the smaller one must
            be interpolated to be able to concatenate them.
    """

    dataset = GetDatasetInformation(os.path.join(DATA_ROOT, frame), filter_type, mode='prediction')

    fix_set = dataset.fix_files
    mov_set = dataset.mov_files
    fix_vols = dataset.fix_vols
    mov_vols = dataset.mov_vols

    dims = dataset.get_biggest_dimensions()

    fix_set = fix_set[PSN - 1]
    mov_set = mov_set[PSN - 1]
    fix_vols = fix_vols[PSN - 1]
    mov_vols = mov_vols[PSN - 1]

    print('Creating prediction patches ... ')

    hdf_data = LoadHDF5File(data_files, fix_set, mov_set,
                            fix_vols, mov_vols, dims)
    hdf_data.normalize()
    hdf_data.to(device)
    
    hdf_data.interpolate_and_concatenate()
    
    vol_data = torch.cat((hdf_data.fix_data, hdf_data.mov_data), 0)

    patched_vol_data, loc = create_patches(vol_data, patch_size, stride, device)

    fixed_patches = patched_vol_data[:, 0, :].to(device)
    moving_patches = patched_vol_data[:, 1, :].to(device)

    print('Finished creating patches')

    return fixed_patches.unsqueeze(1), moving_patches.unsqueeze(1), loc
