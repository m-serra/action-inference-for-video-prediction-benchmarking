from __future__ import absolute_import
import os
from data_readers.bair_predictions_data_reader import BairPredictionsDataReader
from action_inference.savp_vae_gear import SavpVaeGear

"""
export PYTHONPATH="${PYTHONPATH}:~/Tese/action-inference-for-video-prediction-benchmarking"
"""


def main():

    # --> allow seq_len to be None and by default use maximum possible sequence

    model_list = ['savp']

    gear_map = {'savp': SavpVaeGear,
                'savp_vae': SavpVaeGear}

    for model_name in model_list:

        gear_class = gear_map.get(model_name)

        ai = gear_class(model_name=model_name,
                        dataset_name='bair_predictions')

        predictions_save_dir = os.path.join(
                                os.path.expanduser('~/'),
                                'Tese/action-inference-for-video-prediction-benchmarking/datasets/prediction_datasets')

        d_pred = BairPredictionsDataReader(batch_size=8,
                                           sequence_length_train=28,
                                           sequence_length_test=28,
                                           shuffle=True,
                                           dataset_dir=predictions_save_dir,
                                           model_name=model_name)

        ai.train_inference_model(data_reader=d_pred,
                                 n_epochs=10,
                                 seq_len=28,
                                 model_save_dir=os.path.join(os.path.expanduser('~/'),
                                                             'Tese/action-inference-for-video-prediction-benchmarking/',
                                                             'pretrained_inference_models'))


if __name__ == '__main__':
    main()

