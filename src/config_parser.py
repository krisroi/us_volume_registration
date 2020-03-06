import configparser
import os


class UserConfigParser:
    def __init__(self):
        self.config_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                            'main_config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(self.config_filepath)
        self.__parse_content()

    def __parse_content(self):
        self.__parse_network()
        self.__parse_predict()

    def __parse_network(self):
        if self.config.has_option('Network', 'encoder_config'):
            self.encoder_config = tuple(int(x) for x in self.config['Network']['encoder_config'].split('#')[0].split(','))
        else:
            raise AttributeError('Missing parameter [Network][encoder_config]')

        if self.config.has_option('Network', 'growth_rate'):
            self.growth_rate = int(self.config['Network']['growth_rate'].split('#')[0].strip())
        else:
            raise AttributeError('Missing parameter [Network][growth_rate]')

        if self.config.has_option('Network', 'num_init_features'):
            self.num_init_features = int(self.config['Network']['num_init_features'].split('#')[0].strip())
        else:
            raise AttributeError('Missing parameter [Network][num_init_features]')

        if self.config.has_option('Network', 'affine_config'):
            self.affine_config = tuple(int(x) for x in self.config['Network']['affine_config'].split('#')[0].split(','))
        else:
            raise AttributeError('Missing parameter [Network][affine_config]')

        if self.config.has_option('Network', 'num_init_parameters'):
            self.num_init_parameters = int(self.config['Network']['num_init_parameters'].split('#')[0].strip())
        else:
            raise AttributeError('Missing parameter [Network][num_init_parameters]')

    def __parse_predict(self):
        if self.config.has_option('Predict', 'batch_size'):
            self.batch_size = int(self.config['Predict']['batch_size'].split('#')[0].strip())
        else:
            raise AttributeError('Missing parameter [Predict][batch_size]')

        if self.config.has_option('Predict', 'patch_size'):
            self.patch_size = int(self.config['Predict']['patch_size'].split('#')[0].strip())
        else:
            raise AttributeError('Missing parameter [Predict][patch_size]')

        if self.config.has_option('Predict', 'stride'):
            self.stride = int(self.config['Predict']['stride'].split('#')[0].strip())
        else:
            raise AttributeError('Missing parameter [Predict][stride]')


def remove():
    user_config = UserConfigParser()
    if user_config.config.has_section('Predict'):
        user_config.config.remove_section('Predict')
        with open('main_config.ini', 'w') as configWriter:
            user_config.config.write(configWriter)


def write(bs, ps, st):
    write_config = configparser.ConfigParser()
    write_config.add_section('Predict')
    write_config.set('Predict', 'batch_size', str(bs))
    write_config.set('Predict', 'patch_size', str(ps))
    write_config.set('Predict', 'stride', str(st))
    with open('main_config.ini', 'a+') as configWriter:
        write_config.write(configWriter)
