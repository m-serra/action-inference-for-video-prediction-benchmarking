import os
from data_readers.bair_data_reader import BairDataReader
from action_inference.savp_models_gear import SAVPModelsGear
from action_inference.svg_gear import SVGGear
from action_inference.cdna_gear import CDNAGear
from tensorflow.python.keras import backend as K
from training_flags import FLAGS
from utils.make_plots import make_plots_vp_metrics

"""
export PYTHONPATH="${PYTHONPATH}:~/Tese/action-inference-for-video-prediction-benchmarking"
"""


def main():

    # --> allow seq_len to be None and by default use maximum possible sequence
    # --> having an out of memory problem when running pytorch svg with tensorflow models
    # model_list = ['savp_vae', 'savp', 'sv2p', 'cdna']
    model_list = ['svg']

    gear_map = {'savp': SAVPModelsGear,
                'savp_vae': SAVPModelsGear,
                'sv2p': SAVPModelsGear,
                'cdna': CDNAGear,
                'svg': SVGGear}

    for model_name in model_list:

        K.clear_session()
        print('===== ', model_name, ' =====')

        processed_data_dir = None if model_name is not 'svg' else os.path.join(os.path.expanduser('~/'),
                                                                               'data/processed_data')

        d = BairDataReader(dataset_dir=FLAGS.bair_dir,
                           processed_dataset_dir=processed_data_dir,
                           batch_size=8,
                           sequence_length_train=30,
                           sequence_length_test=30,
                           shuffle=True,
                           dataset_repeat=None)

        gear_class = gear_map.get(model_name)

        ai = gear_class(model_name=model_name,
                        dataset_name='bair')

        ai.evaluate_vp_model(data_reader=d,
                             sequence_length=30,
                             context_frames=2,
                             vp_ckpts_dir=os.path.join(os.path.expanduser('~/'),
                                                'Tese/action-inference-for-video-prediction-benchmarking/pretrained_models'),
                             results_dir=os.path.join(os.path.expanduser('~/'),
                                                'Tese/action-inference-for-video-prediction-benchmarking/results'))

    model_list = ['savp_vae', 'savp', 'sv2p', 'cdna', 'svg']
    make_plots_vp_metrics(model_list=model_list,
                          results_dir=os.path.join(os.path.expanduser('~/'),
                                                   'Tese/action-inference-for-video-prediction-benchmarking/results/bair'))


if __name__ == '__main__':
    main()

