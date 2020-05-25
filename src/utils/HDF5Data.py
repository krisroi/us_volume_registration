import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class LoadHDF5File():
    """Loading .h5 files and returns them in separate tensors, one fixed and one moving.
        Args:
            filepath (string): absolute path to .h5 files
            fix_file (string): relative path to specific fixed file
            mov_file (string): relative path to specific moving file
            fix_vol_no (string): specific volume to extract from the fixed file
            mov_vol_no (string): specific volume to extract from the moving file
            dims (tuple): tuple containing the biggest x-, y- and z-dimensions in the dataset
        Returns:
            Two tensors containing fixed and moving data.
            The returned tensors are on the form [1, x-size, y-size, z-size].
        Use example:
            hdf_data = LoadHDF5File(data_files, fix_set, mov_set, fix_vols, mov_vols, dims) #Loads hdf-data
            hdf_data.normalize() #Normalizes hdf-data
            hdf_data.to(device) #Cast hdf-data to specified device

            hdf_data._interpolate() #Interpolate fixed and moving data

            hdf_data.fix_data #Returns fixed hdf_data
            hdf_data.mov_data #Returns moving hdf_data

        Note:
            Tensors should be concatenated before further usage, like:

            hdf_data._concatenate()

            Then the resulting volume data is accesed through

            hdf_data.data
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
            
            del fix_vol
            del mov_vol

        return fix_data, mov_data

    def _interpolate(self):
        '''Scaled interpolation of hdf-data. Interpolates the x-dimension to the biggest x-dimension in the dataset
            while keeping the dimension-ratio the same.
        '''
        self.fix_data = F.interpolate(self.fix_data.unsqueeze(0), scale_factor=(self.dims[0] / self.fix_data.shape[1]),
                                      mode='trilinear', align_corners=False)
        self.mov_data = F.interpolate(self.mov_data.unsqueeze(0), scale_factor=(self.dims[0] / self.mov_data.shape[1]),
                                      mode='trilinear', align_corners=False)
        self.fix_data = self.fix_data.squeeze(1)
        self.mov_data = self.mov_data.squeeze(1)

    def _concatenate(self):
        '''Concatenate fixed- and moving data for further usage.
        '''
        self.data = torch.cat((self.fix_data, self.mov_data), 0)

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
