import torch
import pandas as pd
import os

from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader

from utils.patch_volume import create_patches
from utils.utility_functions import progress_printer
from utils.load_hdf5 import LoadHDF5File


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


class GetDatasetInformation():
    """Reads dataset information from a .txt file and returns the information as lists
        Args:
            filepath (string): filepath to the .txt file
            filename (string): filename of the .txt file
    """

    def __init__(self, filename, filter_type):
        super(GetDatasetInformation, self).__init__()

        self.filename = filename
        self.filter_type = filter_type

        self.fix_files, self.mov_files, self.fix_vols, self.mov_vols = self.load_dataset()

    def load_dataset(self):
        """ Reads the dataset information, pulls out the usable datasets
            and returns them together with corresponding volumes.
        """

        data = pd.read_csv(self.filename)
        data = data.loc[lambda df: data.usable == 'y', :]  # Extract only usable datasets (y: yes)
        ref_filename = data.ref_filename
        mov_filename = data.mov_filename
        # Adding 1 to frame number because volume 0 is called vol01 in file (1 is vol02 etc.)
        ref_vol_frame_no = data.ref_vol_frame_no + 1
        mov_vol_frame_no = data.mov_vol_frame_no + 1

        # Initializing empty list-holders
        fix_files = []
        mov_files = []
        fix_vols = []
        mov_vols = []

        for _, pat_idx in enumerate((ref_filename.index)):
            fix_files.append('{}_{}.h5'.format(ref_filename[pat_idx], self.filter_type))
            mov_files.append('{}_{}.h5'.format(mov_filename[pat_idx], self.filter_type))
            fix_vols.append('{:02}'.format(ref_vol_frame_no[pat_idx]))
            mov_vols.append('{:02}'.format(mov_vol_frame_no[pat_idx]))

        return fix_files, mov_files, fix_vols, mov_vols


def generate_trainPatches(data_information, data_files, filter_type,
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

    dataset = GetDatasetInformation(data_information, filter_type)

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

        vol_data = LoadHDF5File(data_files, fix_set[set_idx], mov_set[set_idx],
                                fix_vols[set_idx], mov_vols[set_idx])
        vol_data.normalize()
        vol_data.to(device)

        patched_vol_data, _ = create_patches(vol_data.data, patch_size, stride, device, voxelsize)
        patched_vol_data = patched_vol_data.cpu()

        fixed_patches = torch.cat((fixed_patches, patched_vol_data[:, 0, :]))
        moving_patches = torch.cat((moving_patches, patched_vol_data[:, 1, :]))

        del patched_vol_data

    print('Finished creating patches')

    shuffled_fixed_patches = torch.zeros((fixed_patches.shape[0], patch_size, patch_size, patch_size)).cpu()
    shuffled_moving_patches = torch.zeros((fixed_patches.shape[0], patch_size, patch_size, patch_size)).cpu()

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

    return shuffled_fixed_patches.unsqueeze(1), shuffled_moving_patches.unsqueeze(1)


def generate_predictionPatches(DATA_ROOT, data_files, filter_type, patch_size, stride, device, voxelsize):
    """Loading all datasets, creates patches and store all patches in a single array.
        Args:
            DATA_ROOT (string): root folder to all data-files
            data_files (string): folder containing the specific .h5 data
            filter_type (string): filter type used for pre-processing
            patch_size (int): desired patch size
            stride (int): desired stride between patches
            voxelsize (float): not used here, but create_patches has it as input
        Returns:
            fixed patches: all fixed patches in the dataset ([num_patches, 1, **patch_size])
            moving patches: all moving patches in the dataset ([num_patches, 1, **patch_size])
            loc: location of each patch
    """

    fix_set = 'J65BP1R0_ecg_{}.h5'.format(filter_type)
    mov_set = 'J65BP1R2_ecg_{}.h5'.format(filter_type)
    fix_vols = '01'
    mov_vols = '12'

    print('Creating patches ... ')

    vol_data = LoadHDF5File(data_files, fix_set, mov_set,
                            fix_vols, mov_vols)
    vol_data.normalize()
    vol_data.to(device)

    patched_vol_data, loc = create_patches(vol_data.data, patch_size, stride, device, voxelsize)

    print("Patched vol_data cuda: ", patched_vol_data.is_cuda)

    return patched_vol_data[:, 0, :].unsqueeze(1), patched_vol_data[:, 1, :].unsqueeze(1), loc
