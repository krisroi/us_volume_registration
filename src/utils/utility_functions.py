import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import math
import torch


def plot_featuremaps(data):
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(20, 10))

    middle_slice = data[0, 0, data.shape[2] - 1]
    ax[0, 0].imshow(middle_slice, origin='left', cmap='gray')
    ax[0, 0].title.set_text('kernel 1')

    plt.show()


def progress_printer(percentage):
    """Function returning a progress bar
        Args:
            percentage (float): percentage point
    """
    eq = '=====================>'
    dots = '......................'
    printer = '[{}{}]'.format(eq[len(eq) - math.ceil(percentage * 20):len(eq)], dots[2:len(eq) - math.ceil(percentage * 20)])
    return printer


# Utility for counting parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Parse string and return boolean value


def parse_boolean(arg):
    if arg.lower() in ('yes', 'true', '1', 'y', 't'):
        return True
    elif arg.lower() in ('no', 'false', '0', 'n', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


# Function for printing information on forward
def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())


# Function for printing information on backward
def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm)


def plotFeatureMaps(input, className):
    num_maps = input.data.shape[1]
    x_range = 10 if num_maps >= 10 else math.ceil(num_maps / 2)
    y_range = math.ceil(num_maps / x_range)
    fig_x, ax_x = plt.subplots(y_range, x_range, squeeze=False, figsize=(40, 40))
    fig_y, ax_y = plt.subplots(y_range, x_range, squeeze=False, figsize=(40, 40))
    fig_z, ax_z = plt.subplots(y_range, x_range, squeeze=False, figsize=(40, 40))
    count = 0
    for i in range(y_range):
        for j in range(x_range):
            ax_x[i, j].get_xaxis().set_visible(False)
            ax_x[i, j].get_yaxis().set_visible(False)
            ax_y[i, j].get_xaxis().set_visible(False)
            ax_y[i, j].get_yaxis().set_visible(False)
            ax_z[i, j].get_xaxis().set_visible(False)
            ax_z[i, j].get_yaxis().set_visible(False)
            if count != num_maps:
                featureMap_x = input.data[0, count, input.data.shape[2] - 1]
                featureMap_y = input.data[0, count, :, input.data.shape[2] - 1]
                featureMap_z = input.data[0, count, :, :, input.data.shape[2] - 1]
                ax_x[i, j].imshow(featureMap_x, origin='left', cmap='gray')
                ax_y[i, j].imshow(featureMap_y, origin='left', cmap='gray')
                ax_z[i, j].imshow(featureMap_z, origin='left', cmap='gray')
                count += 1
    fig_x.suptitle('x-sliced feature maps in class {}'.format(className))
    fig_y.suptitle('y-sliced feature maps in class {}'.format(className))
    fig_z.suptitle('z-sliced feature maps in class {}'.format(className))
    plt.show()


def printFeatureMaps(self, input, output):
    className = self.__class__.__name__
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('Inside class: ' + self.__class__.__name__)
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    #print('input size:', input[0].size())
    print('output size:', output.data.size())
    plotFeatureMaps(output, className)
