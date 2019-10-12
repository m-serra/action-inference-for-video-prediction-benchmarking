from __future__ import absolute_import
import os
import argparse
from data_readers.bair_predictions_data_reader import BairPredictionsDataReader
from action_inference.savp_models_gear import SAVPModelsGear
from action_inference.cdna_gear import CDNAGear
from action_inference.svg_gear import SVGGear

gear_map = {'savp': SAVPModelsGear,
            'savp_vae': SAVPModelsGear,
            'savp_gan': SAVPModelsGear,
            'sv2p': SAVPModelsGear,
            'cdna': CDNAGear,
            'svg': SVGGear}


def main():

    # =============== Parse arguments ===============
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_list", nargs='+', help="Names of the models to use", required=True)
    parser.add_argument("--predictions_data_dir", type=str, help="path to predictions .tfrecord data", required=True)
    parser.add_argument("--model_save_dir", type=str,
                        help="path to the directory where the trained action inference model will be saved",
                        required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs for training")
    parser.add_argument("--sequence_length_train", type=int, default=28, help="number of time steps for training")
    parser.add_argument("--sequence_length_test", type=int, default=28, help="number of time steps for testing")
    parser.add_argument("--train_num_examples", type=int, default=None,
                        help="Number of examples to train on. If None the whole train dataset is used")
    parser.add_argument("--val_num_examples", type=int, default=None,
                        help="Number of examples to validate on. If None the whole val dataset is used")

    args = parser.parse_args()
    model_list = args.model_list
    predictions_data_dir = args.predictions_data_dir
    model_save_dir = args.model_save_dir
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    sequence_length_train = args.sequence_length_train
    sequence_length_test = args.sequence_length_test
    train_num_examples = args.train_num_examples
    val_num_examples = args.val_num_examples

    # =============== Usage ===============
    for model_name in model_list:

        gear_class = gear_map.get(model_name)

        ai = gear_class(model_name=model_name)

        d_pred = BairPredictionsDataReader(dataset_dir=predictions_data_dir,
                                           batch_size=batch_size,
                                           sequence_length_train=sequence_length_train,
                                           sequence_length_test=sequence_length_test,
                                           shuffle=True,
                                           model_name=model_name)

        ai.train_inference_model(data_reader=d_pred,
                                 n_epochs=n_epochs,
                                 model_save_dir=model_save_dir,
                                 train_num_examples=train_num_examples,
                                 val_num_examples=val_num_examples)


if __name__ == '__main__':
    main()
