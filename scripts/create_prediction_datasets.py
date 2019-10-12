from __future__ import absolute_import
import os
import tensorflow as tf
from data_readers.bair_data_reader import BairDataReader
from action_inference.savp_models_gear import SAVPModelsGear
from action_inference.cdna_gear import CDNAGear
from action_inference.svg_gear import SVGGear
import argparse


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
    parser.add_argument("--data_dir", type=str, help="path to .tfrecord data", required=True)
    parser.add_argument("--save_dir", type=str,
                        help="path to the directory where the predictions dataset will be saved",
                        required=True)
    parser.add_argument("--vp_ckpts_dir", type=str,
                        help="path containing a subdirectory 'dataset_name/model_name', which as the vp models checkpoints",
                        required=True)
    parser.add_argument("--batch_size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--context_frames", type=int, default=2, help="number of frames the model receives as input")
    parser.add_argument("--sequence_length_train", type=int, default=30, help="number of time steps for training")
    parser.add_argument("--sequence_length_test", type=int, default=30, help="number of time steps for testing")
    parser.add_argument("--processed_data_dir", type=str, default=None, help="path to processed data (.png)")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Number of examples in the new dataset. If None the whole original dataset is used")

    args = parser.parse_args()
    model_list = args.model_list
    batch_size = args.batch_size
    context_frames = args.context_frames
    sequence_length_train = args.sequence_length_train
    sequence_length_test = args.sequence_length_test
    data_dir = args.data_dir
    _processed_data_dir = args.processed_data_dir
    predictions_save_dir = args.save_dir
    vp_ckpts_dir = args.vp_ckpts_dir
    num_examples = args.num_examples

    # =============== Usage ===============
    for mode in ['train', 'test']:
        print('=====', mode, '=====')
        for model_name in model_list:

            print('=====', model_name, '=====')
            tf.reset_default_graph()

            processed_data_dir = None if model_name is not 'svg' else _processed_data_dir

            d = BairDataReader(dataset_dir=data_dir,
                               processed_dataset_dir=processed_data_dir,
                               batch_size=batch_size,
                               sequence_length_train=sequence_length_train,
                               sequence_length_test=sequence_length_test,
                               shuffle=False,
                               dataset_repeat=None)

            gear_class = gear_map.get(model_name)

            ai = gear_class(model_name=model_name)

            ai.create_predictions_dataset(original_dataset=d,
                                          mode=mode,
                                          context_frames=context_frames,
                                          predictions_save_dir=predictions_save_dir,
                                          vp_ckpts_dir=vp_ckpts_dir,
                                          num_examples=num_examples)


if __name__ == '__main__':
    main()
