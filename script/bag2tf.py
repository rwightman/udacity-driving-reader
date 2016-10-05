# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from __future__ import print_function
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import cv2
import rosbag
import argparse
import tensorflow as tf

LEFT_CAMERA_TOPIC = "/left_camera/image_color"
CENTER_CAMERA_TOPIC = "/center_camera/image_color"
RIGHT_CAMERA_TOPIC = "/right_camera/image_color"
STEERING_TOPIC = "/vehicle/steering_report"


def feature_int64(value_list):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


def feature_float(value_list):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value_list, list):
        value_list = [value_list]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value_list))


def feature_bytes(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def feature_bytes_list(value_list, skip_convert=False):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value_list))


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def to_steering_dict(msg=None):
    if msg is None:
        steering_dict = {
            'steering/timestamp': feature_int64(0),
            'steering/seq': feature_int64(0),
            'steering/angle': feature_float(0.0),
            'steering/torque': feature_float(0.0),
            'steering/speed': feature_float(0.0),
        }    
    else:
        steering_dict = {
            'steering/timestamp': feature_int64(msg.header.stamp.to_nsec()),
            'steering/seq': feature_int64(msg.header.seq),
            'steering/angle': feature_float(msg.steering_wheel_angle),
            'steering/torque': feature_float(msg.steering_wheel_torque),
            'steering/speed': feature_float(msg.speed),
        }
    return steering_dict

def write_example(writer, bridge, image_msg, steering_msg, image_fmt='png'):
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        _, encoded = cv2.imencode('.' + image_fmt, cv_image)
        colorspace = b'RGB'
        channels = 3      
        feature_dict = {
            'image/timestamp': feature_int64(image_msg.header.stamp.to_nsec()),
            'image/frame_id': feature_bytes(image_msg.header.frame_id),
            'image/height': feature_int64(image_msg.height),
            'image/width': feature_int64(image_msg.width),
            'image/channels': feature_int64(channels),
            'image/colorspace': feature_bytes(colorspace),
            'image/format': feature_bytes(image_fmt),
            'image/encoded': feature_bytes(encoded.tobytes()),
            #'image/filename': feature_bytes()
        }
        steering_dict = to_steering_dict(steering_msg)
        feature_dict.update(steering_dict)
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        writer.write(example)

    except CvBridgeError as e:
        print(e)


class ShardWriter():
    def __init__(self, outdir, name, num_entries, num_shards=256):
        self.num_entries = num_entries
        self.outdir = outdir
        self.name = name
        self.num_shards = num_shards
        self.num_entries_per_shard = num_entries // num_shards
        self._writer = None
        self._shard_counter = 0
        self._counter = 0

    def _update_writer(self):
        if not self._writer or self._shard_counter >= self.num_entries_per_shard:
            shard = self._counter // self.num_entries_per_shard
            assert(shard <= self.num_shards)
            output_filename = '%s-%.5d-of-%.5d' % (self.name, shard, self.num_shards)
            output_file = os.path.join(self.outdir, output_filename)
            self._writer = tf.python_io.TFRecordWriter(output_file)
            self._shard_counter = 0

    def write(self, example):
        self._update_writer()
        self._writer.write(example.SerializeToString())
        self._shard_counter += 1
        self._counter += 1
        if not self._counter % 1000:
            print('Processed %d of %d images for %s' % (self._counter, self.num_entries, self.name))
            sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to tensorflow sharded records.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-b', '--bagfile', type=str, nargs='?', default='/data/dataset.bag',
        help='Input bag file')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-n', '--num_images', type=int, nargs='?', default=15212,
        help='Number of images per camera')
    parser.add_argument('-s', '--separate', dest='separate', action='store_true', help='Separate sets per camera')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(separate=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    save_dir = args.outdir
    rosbag_file = args.bagfile
    debug_print = args.debug
    separate_streams = args.separate
    num_images = args.num_images # FIXME detect from bag_info (takes more time)

    bridge = CvBridge()

    filter_topics = [LEFT_CAMERA_TOPIC, CENTER_CAMERA_TOPIC, RIGHT_CAMERA_TOPIC, STEERING_TOPIC]

    if separate_streams:
        left_outdir = get_outdir(save_dir, "left")
        center_outdir = get_outdir(save_dir, "center")
        right_outdir = get_outdir(save_dir, "right")
        shard_writer_left = ShardWriter(left_outdir, 'left', num_images, num_shards=32)
        shard_writer_center = ShardWriter(center_outdir,'center', num_images, num_shards=32)
        shard_writer_right = ShardWriter(right_outdir, 'right', num_images, num_shards=32)
    else:
        single_outdir = get_outdir(save_dir, "combined")
        shard_writer = ShardWriter(single_outdir, 'combined', 3*num_images, num_shards=128)

    latest_steering_msg = None
    example_count = 0
    with rosbag.Bag(rosbag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=filter_topics):
            if (topic == LEFT_CAMERA_TOPIC or 
                topic == CENTER_CAMERA_TOPIC or
                topic == RIGHT_CAMERA_TOPIC):

                if debug_print:
                    print("%s_camera %d" % (msg.header.frame_id[0], msg.header.stamp.to_nsec()))
                    print("steering %d, %f" % 
                        (latest_steering_msg.header.stamp.to_nsec(), latest_steering_msg.steering_wheel_angle))

                if separate_streams:
                    if topic == LEFT_CAMERA_TOPIC:
                        writer = shard_writer_left
                    elif topic == CENTER_CAMERA_TOPIC:
                        writer = shard_writer_center
                    elif topic == RIGHT_CAMERA_TOPIC:
                        writer = shard_writer_right
                else:
                    writer = shard_writer

                write_example(writer, bridge, msg, latest_steering_msg, image_fmt=img_format)
                example_count += 1
                   
            elif topic == STEERING_TOPIC:
                if debug_print:
                    print("steering %d, %f" % (msg.header.stamp.to_nsec(), msg.steering_wheel_angle))

                latest_steering_msg = msg
                
    print("Completed processing %d images to TF examples." % example_count)
    sys.stdout.flush()


if __name__ == '__main__':
    main()