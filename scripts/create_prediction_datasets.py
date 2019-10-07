from __future__ import absolute_import
import os
import tensorflow as tf
from data_readers.bair_data_reader import BairDataReader
from action_inference.savp_models_gear import SAVPModelsGear
from action_inference.cdna_gear import CDNAGear
from action_inference.svg_gear import SVGGear
from training_flags import FLAGS

"""
export PYTHONPATH="${PYTHONPATH}:~/Tese/action-inference-for-video-prediction-benchmarking"
"""


def main():

    mode_list = ['test']
    model_list = ['svg']

    gear_map = {'savp': SAVPModelsGear,
                'savp_vae': SAVPModelsGear,
                'savp_gan': SAVPModelsGear,
                'sv2p': SAVPModelsGear,
                'cdna': CDNAGear,
                'svg': SVGGear}

    # --> allow seq_len to be None and by default use maximum possible sequence
    for mode in mode_list:
        for model_name in model_list:

            print('===== ', model_name, ' =====')
            tf.reset_default_graph()

            processed_data_dir = None if model_name is not 'svg' else os.path.join(os.path.expanduser('~/'),
                                                                                   'data/processed_data')

            d = BairDataReader(dataset_dir=FLAGS.bair_dir,
                               processed_dataset_dir=processed_data_dir,
                               batch_size=8,
                               sequence_length_train=30,
                               sequence_length_test=30,
                               shuffle=True,
                               dataset_repeat=None)  # --> maybe add something for dataset repeat=1 if mode 'test'

            gear_class = gear_map.get(model_name)

            ai = gear_class(model_name=model_name,
                            dataset_name='bair')

            predictions_save_dir = os.path.join(
                                    os.path.expanduser('~/'),
                                    'Tese/action-inference-for-video-prediction-benchmarking/datasets/prediction_datasets')

            ai.create_predictions_dataset(original_dataset=d,
                                          mode=mode,
                                          context_frames=2,
                                          sequence_length=30,
                                          predictions_save_dir=predictions_save_dir,
                                          vp_ckpts_dir=os.path.join(os.path.expanduser('~/'),
                                              'Tese/action-inference-for-video-prediction-benchmarking/pretrained_models'))


if __name__ == '__main__':
    main()
