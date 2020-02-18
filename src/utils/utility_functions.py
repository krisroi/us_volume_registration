import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import math


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
