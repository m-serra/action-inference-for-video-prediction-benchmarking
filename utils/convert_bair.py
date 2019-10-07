import os
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.platform import gfile
from imageio import imsave

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', help='base directory to read data')
parser.add_argument('--save_dir', default='', help='base directory to save processed data')
opt = parser.parse_args()


def get_seq(dname):

    data_dir = '%s/%s' % (opt.data_dir, dname)
    filenames = gfile.Glob(os.path.join(data_dir, '*'))
    if not filenames:
        raise RuntimeError('No data files found.')

    for f in filenames:
        k = 0
        for serialized_example in tf.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            image_seq = []
            action_seq = []
            state_seq = []
            for i in range(30):
                image_name = str(i) + '/image_aux1/encoded'
                action_name = str(i) + '/action'
                state_name = str(i) + '/endeffector_pos'
                byte_str = example.features.feature[image_name].bytes_list.value[0]
                action = example.features.feature[action_name].float_list.value[:]
                state = example.features.feature[state_name].float_list.value[:]
                # img = Image.open(io.BytesIO(byte_str))
                img = Image.frombytes('RGB', (64, 64), byte_str)
                arr = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
                # image_seq.append(arr.reshape(1, 64, 64, 3)/255.)
                image_seq.append(arr.reshape(1, 64, 64, 3).astype('uint8'))
                action_seq.append(action)
                state_seq.append(state)
            image_seq = np.concatenate(image_seq, axis=0)
            action_seq = np.array(action_seq)
            state_seq = np.array(state_seq)
            k = k+1
            yield f, k, image_seq, action_seq, state_seq


def convert_data(dname):    
    seq_generator = get_seq(dname)
    n = 0
    save_dir = opt.save_dir
    while True:
        n += 1
        try:
            f, k, seq, action_seq, state_seq = next(seq_generator)
        except StopIteration:
            break

        f = f.split('/')[-1]

        # the [:-10] removes '.tfrecord' from the name of the file, to create a directory called 'traj_#_to_##'
        os.makedirs('%s/processed_data/%s/%s/%d/' % (save_dir, dname,  f[:-10], k), mode=0o777, exist_ok=True)
       
        for i in range(len(seq)):
            imsave('%s/processed_data/%s/%s/%d/%d.png' % (save_dir, dname,  f[:-10], k, i), seq[i])
        
        with open('%s/processed_data/%s/%s/%d/action.pickle' % (save_dir, dname,  f[:-10], k), 'wb') as file:
            pickle.dump(action_seq, file)

        with open('%s/processed_data/%s/%s/%d/state.pickle' % (save_dir, dname,  f[:-10], k), 'wb') as file:
            pickle.dump(state_seq, file)
        
        print('%s data: %s (%d)  (%d)' % (dname, f, k, n))


if __name__ == '__main__':
    convert_data('train')
    convert_data('test')
