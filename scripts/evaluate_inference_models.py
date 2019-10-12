import os
import argparse
from data_readers.bair_predictions_data_reader import BairPredictionsDataReader
from action_inference.savp_models_gear import SAVPModelsGear
from action_inference.cdna_gear import CDNAGear
from action_inference.svg_gear import SVGGear
from utils.make_plots import make_plots_action_inference
from tensorflow.python.keras import backend as K

gear_map = {'savp': SAVPModelsGear,
            'savp_vae': SAVPModelsGear,
            'sv2p': SAVPModelsGear,
            'cdna': CDNAGear,
            'svg': SVGGear}


def main():
    # =============== Parse arguments ===============
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_list", nargs='+', help="Names of the models to use", required=True)
    parser.add_argument("--predictions_data_dir", type=str, help="path to predictions .tfrecord data", required=True)
    parser.add_argument("--model_ckpt_dir", type=str,
                        help="path to the directory where the trained action inference model is be saved",
                        required=True)
    parser.add_argument("--results_save_dir", type=str,
                        help="path to the directory where the pickle file with metric results will be saved",
                        required=True)
    parser.add_argument("--sequence_length_train", type=int, default=28, help="number of time steps for training")
    parser.add_argument("--sequence_length_test", type=int, default=28, help="number of time steps for testing")
    parser.add_argument("--num_examples", type=int, default=None, help="number of examples to evaluate on")

    args = parser.parse_args()
    model_list = args.model_list
    sequence_length_train = args.sequence_length_train
    sequence_length_test = args.sequence_length_test
    predictions_data_dir = args.predictions_data_dir
    model_ckpt_dir = args.model_ckpt_dir
    results_save_dir = args.results_save_dir
    num_examples = args.num_examples

    for model_name in model_list:

        K.clear_session()
        print('Model: ', model_name)

        gear_class = gear_map.get(model_name)

        ai = gear_class(model_name=model_name)

        d_pred = BairPredictionsDataReader(batch_size=1,
                                           sequence_length_train=sequence_length_train,
                                           sequence_length_test=sequence_length_test,
                                           shuffle=False,
                                           dataset_dir=predictions_data_dir,
                                           model_name=model_name)

        ai.evaluate_inference_model(d_pred, model_ckpt_dir=model_ckpt_dir, results_save_dir=results_save_dir,
                                    num_examples=num_examples)

    make_plots_action_inference(model_list=model_list,
                                results_dir=results_save_dir)


if __name__ == '__main__':
    main()

