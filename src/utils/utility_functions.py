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
    print('grad_input norm:', grad_input[0].norm())
