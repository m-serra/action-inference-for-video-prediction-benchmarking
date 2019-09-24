import os, datetime
from tensorflow.python.platform import flags

DATASET = 'BAIR' # either 'BAIR' or 'GooglePush'
ARCH = 'CDNA'
VAL_INTERVAL = 10
SUMMARY_INTERVAL = 50
SAVE_INTERVAL = 2000
TIMESTAMP = datetime.datetime.now().strftime('%Y-%m-%d:%H-%M-%S')

# tf_record dataset location:
BAIR_DIR = '/media/Data/datasets/bair/softmotion30_44k'
GOOGLE_DIR = '/media/Data/datasets/GooglePush/push/push_train/'

# create log directories
MODEL_DIR = os.path.join(os.path.expanduser('~/'), 'Tese/action-conditioned-video-prediction/pretrained_models')

FLAGS = flags.FLAGS

# DIRECTORIES FLAG
flags.DEFINE_string('model_name', ARCH.lower(), 'name of the model')
flags.DEFINE_string('bair_dir', BAIR_DIR, 'directory containing BAIR train dataset')
flags.DEFINE_string('google_dir', GOOGLE_DIR, 'directory containing Google Push train dataset')
flags.DEFINE_string('output_dir', MODEL_DIR, 'directory for model checkpoints.')

flags.DEFINE_boolean('double_view', False, 'whether to use images from two different perspectives')
flags.DEFINE_boolean('shuffle_data', True, 'whether to shuffle the data files')

# TRAINING FLAGS
flags.DEFINE_string('dataset', DATASET, 'dataset name: BAIR or GooglePush')
flags.DEFINE_integer('n_epochs', 3, 'number of times the dataset is iterated')
flags.DEFINE_integer('n_iterations', None, 'number of training iterations.')
flags.DEFINE_integer('batch_size', 8, 'batch size for training')
flags.DEFINE_float('train_val_split', 0.9,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')
flags.DEFINE_float('learning_rate', 0.001,'the base learning rate of the generator')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')
flags.DEFINE_float('schedsamp_k', 900, 'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_boolean('using_tf_estimator', False, '')

# MODEL FLAGS
flags.DEFINE_string('model', ARCH,
                    'model architecture to use - CDNA, DNA, or STP')
flags.DEFINE_integer('sequence_length_train', 12,
                     'sequence length, including context frames.')
flags.DEFINE_integer('sequence_length_test', 30,
                     'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')
flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_integer('scale_factor', 300, 'Scale factor for action prediction exponential regularization term')
flags.DEFINE_float('K', 0.0000, 'fit vs. regularization tradeoff')

flags.DEFINE_string('f', '', 'kernel')
