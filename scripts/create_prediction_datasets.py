from __future__ import absolute_import
import os
from data_readers.bair_data_reader import BairDataReader
from action_inference.savp_vae_gear import SavpVaeGear
from training_flags import FLAGS

"""
export PYTHONPATH="${PYTHONPATH}:~/Tese/action-inference-for-video-prediction-benchmarking"
"""


def main():

    mode = 'test'
    model_list = ['savp']

    gear_map = {'savp': SavpVaeGear,
                'savp_vae': SavpVaeGear}

    # --> allow seq_len to be None and by default use maximum possible sequence

    for model_name in model_list:

        d = BairDataReader(dataset_dir=FLAGS.bair_dir,
                           batch_size=8,
                           sequence_length_train=30,
                           sequence_length_test=30,
                           shuffle=False,
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
