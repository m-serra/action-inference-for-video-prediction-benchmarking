from __future__ import absolute_import
import os
from data_readers.bair_data_reader import BairDataReader
from data_readers.bair_predictions_data_reader import BairPredictionsDataReader
from action_inference.savp_vae_gear import SavpVaeGear
from training_flags import FLAGS


def main():

    d = BairDataReader(dataset_dir=FLAGS.bair_dir,
                       batch_size=8,
                       sequence_length_train=30,
                       sequence_length_test=30,
                       shuffle=False,
                       dataset_repeat=None)  # maybe add something for dataset repeat=1 if mode 'test'

    ai = SavpVaeGear(model_name='savp_vae',
                     dataset_name='bair',
                     ckpts_dir=os.path.join(os.path.expanduser('~/'),
                                            'Tese/action-inference-for-video-prediction-benchmarking/pretrained_models'))

    predictions_save_dir = os.path.join(os.path.expanduser('~/'),
                                        'Tese/action-inference-for-video-prediction-benchmarking/datasets/prediction_datasets')

    ai.create_predictions_dataset(original_dataset=d,
                                  mode='test',
                                  context_frames=2,
                                  sequence_length=30,
                                  predictions_save_dir=predictions_save_dir)

    #d_pred = BairPredictionsDataReader(batch_size=8,
    #                                   sequence_length_train=30,
    #                                   sequence_length_test=30,
    #                                  shuffle=True,
    #                                   dataset_dir=predictions_save_dir)

    #ai.train_inference_model(datareader=d_pred,
    #                         n_epochs=3)

if __name__ == '__main__':
    main()

