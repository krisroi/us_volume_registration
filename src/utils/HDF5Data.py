import h5py
import torch
import torch.nn as nn
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

    def __init__(self, filepath, fix_file, mov_file, fix_vol_no, mov_vol_no):
        super(LoadHDF5File, self).__init__()
        self.filepath = filepath
        self.fix_file = fix_file
        self.mov_file = mov_file
        self.fix_vol_no = 'vol{}'.format(fix_vol_no)
        self.mov_vol_no = 'vol{}'.format(mov_vol_no)

        self.data = self.load_hdf5()

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

            shape = list(fix_vol.shape)
            shape = (2, shape[0], shape[1], shape[2])

            vol_data = torch.empty(shape)
            vol_data[0] = torch.from_numpy(fix_vol).float()
            vol_data[1] = torch.from_numpy(mov_vol).float()

        return vol_data

    def normalize(self):
        """ Normalizes pixel data in the .h5 files
        Example:
            volume_data = HDF5Image(required_parameters) # Loading files into volume_data
            volume_data.normalize() # Normalizes the values stored in volume_data
        """
        self.data = torch.div(self.data, torch.max(self.data))

    def to(self, device):
        """ Transfers data-variable to specified device
            Args:
                device (torch.device): desired device
        """
        self.data = self.data.to(device)

    def cpu(self):
        """ Transfers data-variable to CPU
        """
        self.data = self.data.cpu()


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
