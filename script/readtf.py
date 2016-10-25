# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import print_function
import os
import sys
import tensorflow as tf


def datafiles(search_dir, name):
    tf_record_pattern = os.path.join(search_dir, '%s-*' % name)
    data_files = tf.gfile.Glob(tf_record_pattern)
    data_files = sorted(data_files)
    if not data_files:
      print('No files found for dataset %s at %s' % (name, search_dir))
    return data_files


def example_parser(example_serialized):
    
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/timestamp': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
        'steering/angle': tf.FixedLenFeature([1], dtype=tf.float32, default_value=0.0),
        'steering/timestamp': tf.FixedLenFeature([1], dtype=tf.int64, default_value=-1),
    }

    features = tf.parse_single_example(example_serialized, feature_map)

    image_timestamp = tf.cast(features['image/timestamp'], dtype=tf.int64)
    steering_angle = tf.cast(features['steering/angle'], dtype=tf.float32)
    steering_timestamp = tf.cast(features['steering/timestamp'], dtype=tf.int64)

    return features['image/encoded'], image_timestamp, steering_angle, steering_timestamp


def create_read_graph(data_dir, name, num_readers=4, estimated_examples_per_shard=64, coder=None):
    # Get sharded tf example files for the dataset
    data_files = datafiles(data_dir, name)

    # Create queue for sharded tf example files
    # FIXME the num_epochs argument seems to have no impact? Queue keeps looping forever if not stopped.
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1, num_epochs=1)

    # Create queue for examples
    examples_queue = tf.FIFOQueue(capacity=estimated_examples_per_shard + 4, dtypes=[tf.string])

    enqueue_ops = []
    for _ in range(num_readers):
        reader = tf.TFRecordReader()
        _, example = reader.read(filename_queue)
        enqueue_ops.append(examples_queue.enqueue([example]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
    example_serialized = examples_queue.dequeue()
    
    processed = []
    for _ in range(num_readers):
        image_buffer, image_timestamp, steering_angle, steering_timestamp = example_parser(example_serialized)
        decoded_image = tf.image.decode_jpeg(image_buffer)
        processed.append([decoded_image, image_timestamp, steering_angle, steering_timestamp])

    return processed #decoded_image, image_timestamp, steering_angle, steering_timestamp


def main():
    data_dir = '/output/combined'
    num_images = 1452601

    # Build graph and initialize variables
    read_op = create_read_graph(data_dir, 'combined')
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess = tf.Session()
    sess.run(init_op)

    # Start input enqueue threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    read_count = 0
    try:
        while read_count < num_images and not coord.should_stop():
            read_output = sess.run(read_op)
            for o in read_output:
                decoded_image = o[0]
                assert len(decoded_image.shape) == 3
                assert decoded_image.shape[2] == 3
            read_count += len(read_output)
            if not read_count % 1000:
                print("Read %d examples" % read_count)

    except tf.errors.OutOfRangeError:
        print("Reading stopped by Queue")
    finally:
        # Ask the threads to stop.
        coord.request_stop()

    print("Done reading %d images" % read_count)

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()