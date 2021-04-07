import PIL
import os
import numpy as np
import argparse
import lmdb
import sys
import cv2
sys.path.append(".")
from data.image_folder import make_dataset


def create_lmdb_from_images(opt):
    paths = sorted(make_dataset(opt.input))

    output_dir = opt.output
    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # initialize lmdb
    output_dir = opt.output
    lmdb_env = lmdb.open(output_dir, map_size=1099511627776)

    with lmdb_env.begin(write=True) as txn:
        for idx, image_path in enumerate(paths):
            if idx % 10 == 0:
                print('%d\r' % idx, end='', flush=True)
            img = PIL.Image.open(image_path).convert('RGB')
            img = np.asarray(img)
            img = cv2.imencode('.png', img)[1].tostring()
            txn.put(image_path.encode('ascii'), img)


def create_lmdb_from_tfrecords(opt):
    #  initialize tensorflow
    assert opt.stylegan_codebase_path is not None
    sys.path.append(opt.stylegan_codebase_path)
    import dnnlib.tflib as tflib
    from training import dataset
    import tensorflow as tf
    tfrecord_dir = opt.input
    print('Loading dataset "%s"' % tfrecord_dir)
    tflib.init_tf({'gpu_options.allow_growth': True})
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0)
    tflib.init_uninitialized_vars()

    # initialize lmdb
    output_dir = opt.output
    lmdb_env = lmdb.open(output_dir, map_size=1099511627776)

    print('Extracting images to "%s"' % output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    idx = 0
    with lmdb_env.begin(write=True) as txn:
        while True:
            idx += 1
            if idx % 10 == 0:
                print('%d\r' % idx, end='', flush=True)
            try:
                images, _labels = dset.get_minibatch_np(1)
            except tf.errors.OutOfRangeError:
                break
            if images.shape[1] == 1:
                img = PIL.Image.fromarray(images[0][0], 'L').convert('RGB')
            else:
                img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
            img = np.asarray(img)
            img = cv2.imencode('.png', img)[1].tostring()
            imagekey = "%08d" % idx
            txn.put(imagekey.encode('ascii'), img)


        


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser()

    parser.add_argument("mode",
                        choices=("create_lmdb_from_images",
                                 "create_lmdb_from_tfrecords",
                        ))
    parser.add_argument("--input", help="input path")
    parser.add_argument("--output", help="input path")
    parser.add_argument("--stylegan_codebase_path",
                        help="path to stylegan codebase. Path to git clone https://github.com/NVlabs/stylegan.git")

    opt = parser.parse_args()

    if opt.mode == "create_lmdb_from_images":
        create_lmdb_from_images(opt)
    elif opt.mode == "create_lmdb_from_tfrecords":
        create_lmdb_from_tfrecords(opt)

    print("Finished")

