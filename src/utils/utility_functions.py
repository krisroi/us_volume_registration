import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt

def plot_featuremaps(data):
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(20, 10))

    middle_slice = data[0, 0, data.shape[2]-1]
    ax[0, 0].imshow(middle_slice, origin='left', cmap='gray')
    ax[0, 0].title.set_text('kernel 1')

    plt.show()