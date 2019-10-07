import os
from data_readers.bair_predictions_data_reader import BairPredictionsDataReader
from action_inference.savp_models_gear import SAVPModelsGear
from action_inference.cdna_gear import CDNAGear
from action_inference.svg_gear import SVGGear
from utils.make_plots import make_plots_action_inference
from tensorflow.python.keras import backend as K


"""
export PYTHONPATH="${PYTHONPATH}:~/Tese/action-inference-for-video-prediction-benchmarking"
"""


def main():

    # --> allow seq_len to be None and by default use maximum possible sequence
    model_list = ['savp_vae', 'savp', 'sv2p', 'cdna', 'svg']

    gear_map = {'savp': SAVPModelsGear,
                'savp_vae': SAVPModelsGear,
                'sv2p': SAVPModelsGear,
                'cdna': CDNAGear,
                'svg': SVGGear}

    for model_name in model_list:

        K.clear_session()
        print('Model: ', model_name)

        gear_class = gear_map.get(model_name)

        ai = gear_class(model_name=model_name,
                        dataset_name='bair')

        predictions_save_dir = os.path.join(os.path.expanduser('~/'),
                                            'Tese/action-inference-for-video-prediction-benchmarking/datasets/prediction_datasets')

        d_pred = BairPredictionsDataReader(batch_size=1,
                                           sequence_length_train=28,
                                           sequence_length_test=28,
                                           shuffle=False,
                                           dataset_dir=predictions_save_dir,
                                           model_name=model_name)

        ai.evaluate_inference_model(d_pred,
                                    seq_len=28,
                                    model_ckpt_dir=os.path.join(os.path.expanduser('~/'),
                                                'Tese/action-inference-for-video-prediction-benchmarking/pretrained_inference_models'),
                                    results_save_dir=os.path.join(os.path.expanduser('~/'),
                                                'Tese/action-inference-for-video-prediction-benchmarking/results'))

    make_plots_action_inference(model_list=model_list,
                                results_dir=os.path.join(os.path.expanduser('~/'),
                                            'Tese/action-inference-for-video-prediction-benchmarking/results/bair_predictions'))


if __name__ == '__main__':
    main()

