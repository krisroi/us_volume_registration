import argparse


def str2bool(arg):
    """Parse string and return boolean value"""
    if arg.lower() in ('yes', 'true', '1', 'y', 't'):
        return True
    elif arg.lower() in ('no', 'false', '0', 'n', 'f'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')


def float_type(arg):
    """Float-type function for argparse"""
    try:
        f = float(arg)
    except ValueError:
        raise argpase.ArgumentTypeError('Must be a floating point number')
    return f


def int_type(arg):
    """Int-type function for argparse"""
    try:
        i = int(arg)
    except ValueError:
        raise argparse.ArgumentTypeError('Must be an integer')
    return i


def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    f = float_type(arg)
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f


def range_limited_int_type_TOT_NUM_SETS(arg):
    """Type function for argparse - an integer within some predefined bounds"""
    i = int_type(arg)
    if i < 1 or i > 23:
        raise argparse.ArgumentTypeError('Total number of sets must be between {} and {}'.format(1, 23))
    return i


def get_frame(arg):
    """Function for returning correct info-file for given frame"""
    try:
        frame = str(arg)
        if arg.lower() in ('es', 'endsystole', 'end-systole'):
            frame = 'end_systole.csv'
        elif arg.lower() in ('ed', 'enddiastole', 'end-diastole'):
            frame = 'end_diastole.csv'
        else:
            raise argparse.ArgumentTypeError('Specify frame with ES or ED')
    except ValueError:
        raise argparse.ArgumentTypeError('Must be a string')
    return frame


def cross_validation_folds(arg):
    """Return correct number of folds for training"""
    if arg.lower() in ('yes', 'true', '1', 'y', 't'):
        return int(5)
    elif arg.lower() in ('no', 'false', '0', 'n', 'f'):
        return int(1)
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
