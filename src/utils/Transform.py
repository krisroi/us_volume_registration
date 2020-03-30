import SimpleITK as sitk
import numbers
import os
import numpy as np
import torch
from abc import ABC, abstractmethod
from utils.affine_transform import affine_transform


class Transform(ABC):
    """Abstract class for all transforms. Based on TorchIO <https://github.com/fepegar/torchio>.
    All classes used to transform a sample should subclass it.
    All subclasses should overwrite Transform.apply_transform`,
    which takes a sample, applies some transformation and returns the result.
    """

    def __call__(self, sample):
        """Transform a sample and return the result."""
        sample = self.apply_transform(sample)
        return sample

    @abstractmethod
    def apply_transform(self, sample):
        raise NotImplementedError


class RandomAffine(Transform):
    """Transform class that applies a random affine transformation to a sample.
        Args:
            degrees (tuple, int): range of degrees to allow the transformation to use
            translation (tuple, int): range of translation-values to allow the transformation to use
            voxelsize (float)
        Note:
            translation values are in voxels. translation=(1, 3) yields a range of 1 to 3 voxels for translation.
            To get translation in pixels, the values are multiplied with the voxelsize.
    """
    def __init__(self, degrees, translation, voxelsize):
        super().__init__()

        self.degrees = self.parse_degrees(degrees)
        self.translation = self.parse_translation(translation)
        self.voxelsize = voxelsize

    def __call__(self, sample):
        return super().__call__(sample)

    def apply_transform(self, patches):
        transformed_patches = torch.Tensor(patches.shape).cpu()
        for patch_idx in range(patches.shape[0]):
            rotation_params, translation_params = self.get_params(self.degrees, self.translation)
            transformed_patches[patch_idx] = self.apply_affine_transform(patches[patch_idx].unsqueeze(0),
                                                                         rotation_params, translation_params)
        return transformed_patches

    def parse_range(self, nums_range, name):
        if isinstance(nums_range, numbers.Number):
            if nums_range < 0:
                raise ValueError(
                    f'If {name} is a single number,'
                    f' it must be positive, not {nums_range}')
            return (-nums_range, nums_range)
        else:
            if len(nums_range) != 2:
                raise ValueError(
                    f'If {name} is a sequence,'
                    f' it must be of len 2, not {nums_range}')
            min_degree, max_degree = nums_range
            if min_degree > max_degree:
                raise ValueError(
                    f'If {name} is a sequence, the second value must be'
                    f' equal or greater than the first, not {nums_range}')
        return nums_range

    def parse_degrees(self, degrees):
        return self.parse_range(degrees, 'degrees')

    def parse_translation(self, translation):
        return self.parse_range(translation, 'translation')

    def get_params(self, degrees, translation):
        rotation_params = torch.FloatTensor(3).uniform_(*degrees)
        translation_params = torch.FloatTensor(3).uniform_(*translation)
        return rotation_params.tolist(), translation_params.tolist()

    def get_transform(self, degrees, translation):
        transform = sitk.Euler3DTransform()
        radians = np.radians(degrees)
        transform.SetRotation(*radians)
        transform.SetTranslation(translation)
        return transform

    def apply_affine_transform(self, patch, rotation_params, translation_params):
        transform = self.get_transform(rotation_params, translation_params)
        A = torch.Tensor(transform.GetMatrix()).view(-1, 3, 3)
        b = torch.Tensor(transform.GetTranslation()).view(-1, 3, 1)
        theta = torch.cat((A, b * self.voxelsize), dim=2)
        transformed_patch = affine_transform(patch, theta)
        return transformed_patch

