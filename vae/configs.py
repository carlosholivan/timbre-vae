# model parameters

class Config:
    pass


class InputsConfig(Config):

    SAMPLING_RATE           =       16000                   #  Hz
    WINDOW_SIZE             =       1024                    # samples
    HOP_LENGTH              =       512                     #  samples
    F_MIN                   =       32.70                   #  minimum frequency in Hz
    BINS                    =       190
    BINS_PER_OCTAVE         =       24


class ParamsConfig(Config):

    DATA_PATH               =       './data'
    TRAINED_MODELS_PATH     =       './trained_models'      #  directory where to store weigths during training

    BATCH_SIZE              =       1
    NUM_CHANNELS            =       64                      #  output channels after first convolution
    LEARNING_RATE           =       1e-3
    NUM_EPOCHS              =       100

    VAE_BETA                =       1
    LATENT_DIMS             =       3

    LOG_INTERVAL            =       10
    SEED                    =       199


class PlotsConfig(Config):

    PLOTS_PATH              =       './plots'
    COLORS                  =       {'violin'        : '#aed6f1',
                                     'viola'         : '#5499c7',
                                     'cello'         : '#2e86c1',
                                     'contrabass'    : '#1f618d',
                                     'double-bass'   : '#1b4f72',

                                     'clarinet'      : '#a3e4d7',
                                     'bass-clarinet' : '#73c6b6',
                                     'saxophone'     : '#45b39d',
                                     'flute'         : '#148f77',
                                     'oboe'          : '#117864',
                                     'bassoon'       : '#0b5345',

                                     'cor-anglais'   : '#edbb99',
                                     'french-horn'   : '#e59866',
                                     'trombone'      : '#e67e22',
                                     'trumpet'       : '#ba4a00',
                                     'tuba'          : '#784212',

                                     'guitar'        : '#bb8fce',
                                     'mandolin'      : '#9b59b6',
                                     'banjo'         :'#6c3483'
                                    }