import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class LoadHDF5File():
    """Loading .h5 files and returns them in a 2-channel tensor, one fixed- and one moving channel.
        Args:
            filepath (string): absolute path to .h5 files
            fix_file (string): relative path to specific fixed file
            mov_file (string): relative path to specific moving file
            fix_vol_no (string): specific volume to extract from the fixed file
            mov_vol_no (string): specific volume to extract from the moving file
        Returns:
            A tensor that contains both a fixed- and a moving image.
            The returned tensor is on the form [2, x-size, y-size, z-size].
            volume_data.data[0, :] returns the fixed image.
            volume_data.data[1, :] returns the moving image.
    """

    def __init__(self, filepath, fix_file, mov_file, fix_vol_no, mov_vol_no, dims):
        super(LoadHDF5File, self).__init__()
        self.filepath = filepath
        self.fix_file = fix_file
        self.mov_file = mov_file
        self.fix_vol_no = 'vol{}'.format(fix_vol_no)
        self.mov_vol_no = 'vol{}'.format(mov_vol_no)
        self.dims = dims

        self.fix_data, self.mov_data = self.load_hdf5()

        self.data = torch.Tensor([])

    def load_hdf5(self):
        """ Loads HDF5-data from the specified filepath
        """

        fixed = '{}{}'.format(self.filepath, self.fix_file)
        moving = '{}{}'.format(self.filepath, self.mov_file)

        with h5py.File(fixed, 'r') as fix, h5py.File(moving, 'r') as mov:

            fixed_volumes = fix['CartesianVolumes']
            moving_volumes = mov['CartesianVolumes']

            fix_vol = fixed_volumes[self.fix_vol_no][:]
            mov_vol = moving_volumes[self.mov_vol_no][:]

            fix_data = torch.Tensor(1, *fix_vol.shape)
            mov_data = torch.Tensor(1, *mov_vol.shape)

            fix_data[0] = torch.from_numpy(fix_vol).float()
            mov_data[0] = torch.from_numpy(mov_vol).float()

        return fix_data, mov_data

    def interpolate_and_concatenate(self):
        fix_interpolated = F.interpolate(self.fix_data.unsqueeze(0), scale_factor=(self.dims[0]/self.fix_data.shape[1]), 
                                         mode='trilinear', align_corners=False)
        mov_interpolated = F.interpolate(self.mov_data.unsqueeze(0), scale_factor=(self.dims[0]/self.mov_data.shape[1]), 
                                         mode='trilinear', align_corners=False)

        self.data = torch.cat((fix_interpolated.squeeze(0), mov_interpolated.squeeze(0)), 0)

    def normalize(self):
        """ Normalizes pixel data in the .h5 files
        Example:
            volume_data = HDF5Image(required_parameters) # Loading files into volume_data
            volume_data.normalize() # Normalizes the values stored in volume_data
        """
        self.fix_data = torch.div(self.fix_data, torch.max(self.fix_data))
        self.mov_data = torch.div(self.mov_data, torch.max(self.mov_data))

    def to(self, device):
        """ Transfers data-variable to specified device
            Args:
                device (torch.device): desired device
        """
        self.fix_data = self.fix_data.to(device)
        self.mov_data = self.mov_data.to(device)

    def cpu(self):
        """ Transfers data-variables to CPU
        """
        self.fix_data = self.fix_data.cpu()
        self.mov_data = self.mov_data.cpu()


class SaveHDF5File():
    r"""Creates and saves .h5 data. It either creates a new file or truncates the existing one.
        Args:
            DATA_ROOT (string): root folder for data storage
    """

    def __init__(self, DATA_ROOT):
        super(SaveHDF5File, self).__init__()

        self.savedir = os.path.join(DATA_ROOT, 'patch_prediction')
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.filename = os.path.join(self.savedir, 'patch_prediction.h5')

        with h5py.File(self.filename, 'w') as pWrite:
            fVgrp = pWrite.create_group('FixedVolume')
            mVgrp = pWrite.create_group('MovingVolume')
            wVgrp = pWrite.create_group('WarpedVolume')

    def save_hdf5(self, fixed_batch, moving_batch, warped_batch, sampleNumber):
        r"""Saves .h5 data into an already existing .h5 file.
            Args:
                fixed_batch (Tensor): batch containing fixed patches
                moving_batch (Tensor): batch containing moving patches
                warped_batch (Tensor): batch containing warped patcehs
                sampleNumber (int): number to control the dataset-name of the current patch
            Example:
                saveData = SaveHDF5File(DATA_ROOT)
                saveData.save_hdf5(fixed_batch, moving_batch, warped_batch, sampleNumber)
        """
        batch_size = fixed_batch.shape[0]

        for sample_idx in range(batch_size):
            fixed_vol_data = fixed_batch[sample_idx, 0, :].cpu()
            moving_vol_data = moving_batch[sample_idx, 0, :].cpu()
            warped_vol_data = warped_batch[sample_idx, 0, :].cpu()

            fixed_vol_data = self.rescale(fixed_vol_data).int().numpy()
            moving_vol_data = self.rescale(moving_vol_data).int().numpy()
            warped_vol_data = self.rescale(warped_vol_data).int().numpy()

            with h5py.File(self.filename, 'a') as pWrite:
                pWrite['FixedVolume'].create_dataset('patch{:02}'.format(sample_idx + sampleNumber),
                                                     data=fixed_vol_data)
                pWrite['MovingVolume'].create_dataset('patch{:02}'.format(sample_idx + sampleNumber),
                                                      data=moving_vol_data)
                pWrite['WarpedVolume'].create_dataset('patch{:02}'.format(sample_idx + sampleNumber),
                                                      data=warped_vol_data)

        print('Finished saving current batch of patches.')

    def rescale(self, data):
        maxVal = 255
        return maxVal * data
