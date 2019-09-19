from training_flags import FLAGS
from data_readers.bair_data_reader import BairDataReader
from action_inference.base_action_inference_gear import BaseActionInference

def main():

    d = BairDataReader(dataset_dir=FLAGS.bair_dir,
                       batch_size=8,
                       sequence_length_train=28,
                       sequence_length_test=28,
                       shuffle=False,
                       dataset_repeat=None)

    ai = BaseActionInference(model_name='savp_vae',
                             dataset_name='bair')




if __name__ == '__main__':
    main()

