import matplotlib
matplotlib.use('tkagg')

import os
import torch
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import utils.parse_utils as pu
from utils.affine_transform import affine_transform
from utils.HDF5Data import LoadHDF5File
from utils.data import GetDatasetInformation
from losses.ncc_loss import masked_normalized_cross_correlation, unmasked_normalized_cross_correlation
from config_parser import UserConfigParser


def parse():
    parser = argparse.ArgumentParser(description='Align volumes and plot the predicted alignment')
    parser.add_argument('-glob',
                        type=str, required=True,
                        help='Filename of the file containing global theta.')
    parser.add_argument('-frame',
                        type=pu.get_frame, required=True,
                        help='Choose end-diastolic (ED) frame or end-systolic (ES) frame')
    parser.add_argument('-PSN',
                        type=pu.int_type, required=True,
                        help='Specify prediction set number. Available sets: 1, 2, 3')
    parser.add_argument('-aft',
                        choices={"Bilateral_lookup", "NLMF_lookup", "raw"}, default="Bilateral_lookup",
                        help='Specify the filter-type of the files to align')
    parser.add_argument('-save',
                        type=str, default=None,
                        help='Saves the algined volumes with the specified filename. PLEASE INCLUDE FILE EXTENSION')
    parser.add_argument('-calph',
                        type=pu.float_type, default=1.0,
                        help='Specify opacity of the fixed volume. 1.0 = no opacity')
    parser.add_argument('-galph',
                        type=pu.float_type, default=0.3,
                        help='Specify opacity of the moving volume. 1.0 = no opacity')
    args = parser.parse_args()

    return args


def main():
    global args, user_config

    args = parse()
    user_config = UserConfigParser()

    dataset = GetDatasetInformation(os.path.join(user_config.DATA_ROOT, args.frame), args.aft, mode='prediction')

    fixed_image = dataset.fix_files
    moving_image = dataset.mov_files
    fix_vol = dataset.fix_vols
    mov_vol = dataset.mov_vols

    dims = dataset.get_biggest_dimensions()

    voxelsize = 7.000003e-4

    data_files = os.path.join(user_config.DATA_ROOT, 'patient_data_proc_{}/'.format(args.aft))
    theta_proc_path = os.path.join(user_config.PROJECT_ROOT, user_config.PROCRUSTES, 'results', args.glob)

    vol_data = LoadHDF5File(data_files,
                            fixed_image[args.PSN - 1], moving_image[args.PSN - 1],
                            fix_vol[args.PSN - 1], mov_vol[args.PSN - 1], dims=None)
    
    vol_data.normalize()

    fixed_volume = vol_data.fix_data.unsqueeze(0)
    moving_volume = vol_data.mov_data.unsqueeze(0)

    # Reading global theta from file
    global_theta = []
    with open(theta_proc_path, 'r') as readTheta:
        for i, theta in enumerate(readTheta.read().split()):
            if theta != '1' and theta != '0':
                if i == 3 or i == 7 or i == 11:
                    global_theta.append(float(theta) * voxelsize * 10)
                else:
                    global_theta.append(float(theta))

    global_theta = torch.Tensor(global_theta)
    global_theta = global_theta.view(-1, 3, 4)  # Get theta on correct form for affine transform
    print('Global theta:')
    print(global_theta)
    print('\n')

    warped_volume = affine_transform(moving_volume, global_theta)

    pre_loss, pre_mask = unmasked_normalized_cross_correlation(fixed_volume, moving_volume, reduction=None)
    post_loss, post_mask = masked_normalized_cross_correlation(fixed_volume, warped_volume, reduction=None)

    print('\n')
    print('{}'.format('Post alignment values'))
    print('*' * 100)
    print('Prediction set' + ' | ' + 'NCC before warping' + ' | ' + 'NCC after warping' + ' | ' +
          'Improvement' + ' | ' + 'Percentwice imp.')
    print('{:<8}{:>20}{:>20}{:>18}{:>13}%'.format(args.PSN,
                                                  round(pre_loss.item(), 4),
                                                  round(post_loss.item(), 4),
                                                  round((post_loss.item() - pre_loss.item()), 4),
                                                  round(100 - ((pre_loss.item() / post_loss.item()) * 100), 2)))

    plot_volumes(fixed_volume, moving_volume, warped_volume, pre_mask, post_mask)


def plot_volumes(fixed_volume, moving_volume, warped_volume, orig_mask, sect_mask):

    x_range = 3
    y_range = 2

    fig, ax = plt.subplots(y_range, x_range, squeeze=False, figsize=(40, 40))

    fixed_x = fixed_volume[0, 0, fixed_volume.shape[2] // 2]
    fixed_y = fixed_volume[0, 0, :, fixed_volume.shape[3] // 2]
    fixed_z = fixed_volume[0, 0, :, :, fixed_volume.shape[4] // 2]

    moving_x = moving_volume[0, 0, moving_volume.shape[2] // 2]
    moving_y = moving_volume[0, 0, :, moving_volume.shape[3] // 2]
    moving_z = moving_volume[0, 0, :, :, moving_volume.shape[4] // 2]

    warped_x = warped_volume[0, 0, warped_volume.shape[2] // 2]
    warped_y = warped_volume[0, 0, :, warped_volume.shape[3] // 2]
    warped_z = warped_volume[0, 0, :, :, warped_volume.shape[4] // 2]

    orig_mask_x = orig_mask[0, 0, orig_mask.shape[2] // 2]
    orig_mask_y = orig_mask[0, 0, :, orig_mask.shape[3] // 2]
    orig_mask_z = orig_mask[0, 0, :, :, orig_mask.shape[4] // 2]

    sect_mask_x = sect_mask[0, 0, sect_mask.shape[2] // 2]
    sect_mask_y = sect_mask[0, 0, :, sect_mask.shape[3] // 2]
    sect_mask_z = sect_mask[0, 0, :, :, sect_mask.shape[4] // 2]

    for i in range(y_range):
        for j in range(x_range):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)

            ax[i, 0].set_xlim([0, fixed_x.shape[1]])
            ax[i, 0].set_ylim([fixed_x.shape[0], 0])
            ax[i, 1].set_xlim([0, fixed_y.shape[0]])
            ax[i, 1].set_ylim([fixed_y.shape[0], 0])
            ax[i, 2].set_xlim([0, fixed_x.shape[0]])
            ax[i, 2].set_ylim([fixed_x.shape[1], 0])

            #ax[0, j].title.set_text('No alignment')
            #ax[1, j].title.set_text('Predicted alignment')

        ax[i, 0].imshow(fixed_x, origin='left', cmap='copper', alpha=args.calph)
        ax[0, 0].imshow(moving_x, origin='left', cmap='gray', alpha=args.galph)
        ax[1, 0].imshow(warped_x, origin='left', cmap='gray', alpha=args.galph+0.2)

        ax[i, 1].imshow(fixed_y, origin='left', cmap='copper', alpha=args.calph)
        ax[0, 1].imshow(moving_y, origin='left', cmap='gray', alpha=args.galph)
        ax[1, 1].imshow(warped_y, origin='left', cmap='gray', alpha=args.galph+0.2)

        ax[i, 2].imshow(fixed_z, origin='left', cmap='copper', alpha=args.calph)
        ax[0, 2].imshow(moving_z, origin='left', cmap='gray', alpha=args.galph)
        ax[1, 2].imshow(warped_z, origin='left', cmap='gray', alpha=args.galph+0.2)

        #ax[0, 0].imshow(orig_mask_x, origin='left', cmap='cool', alpha=0.1)
        #ax[0, 1].imshow(orig_mask_y, origin='left', cmap='cool', alpha=0.1)
        #ax[0, 2].imshow(orig_mask_z, origin='left', cmap='cool', alpha=0.1)

        #ax[1, 0].imshow(sect_mask_x, origin='left', cmap='cool', alpha=0.1)
        #ax[1, 1].imshow(sect_mask_y, origin='left', cmap='cool', alpha=0.1)
        #ax[1, 2].imshow(sect_mask_z, origin='left', cmap='cool', alpha=0.1)

        #save_data(fig, ax)
        
        
    if args.save is not None:
        output_dir = os.path.join(user_config.PROJECT_ROOT, user_config.PROJECT_NAME, 'output', 'figures')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, args.save), dpi=200,
                    format=args.save.split('.')[1].strip(), bbox_inches='tight', pad_inches=0)
        print('Saved figure at ' + output_dir + ' with filename ' + args.save_filename)

    plt.show()
    
def save_data(fig, ax):
    output_dir = os.path.join(user_config.PROJECT_ROOT, user_config.PROJECT_NAME, 'output', 'figures', 'post_align')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(3):
        if i == 0:
            view = 'lax'
        elif i == 1:
            view = 'shrt'
        else:
            view = 'laz'
        bbox = ax[1, i].get_tightbbox(fig.canvas.get_renderer())
        fig.savefig(os.path.join(output_dir, '{}_pred{}_{}.png'.format(view, args.PSN, 'ES')),
                                dpi = 200, format='png', bbox_inches=bbox.transformed(fig.dpi_scale_trans.inverted()),
                                pad_inches=0)

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')

    main()
