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
import imghdr
import heapq
import argparse
import numpy as np
import tensorflow as tf

from bagutils import *


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


def check_image_format(data):
    img_fmt = imghdr.what(None, h=data)
    return 'jpg' if img_fmt == 'jpeg' else img_fmt


def to_steering_dict(msg=None):
    steering_dict = {
        'steer/timestamp': feature_int64(0),
        'steer/seq': feature_int64(0),
        'steer/angle': feature_float(0.0),
        'steer/torque': feature_float(0.0),
        'steer/speed': feature_float(0.0),
    } if msg is None else {
        'steer/timestamp': feature_int64(msg.header.stamp.to_nsec()),
        'steer/seq': feature_int64(msg.header.seq),
        'steer/angle': feature_float(msg.steering_wheel_angle),
        'steer/torque': feature_float(msg.steering_wheel_torque),
        'steer/speed': feature_float(msg.speed),
    }
    return steering_dict


def to_gps_dict(msg=None):
    gps_dict = {
        'gps/timestamp': feature_int64(0),
        'gps/seq': feature_int64(0),
        'gps/lat': feature_float(0.0),
        'gps/long': feature_float(0.0),
        'gps/alt': feature_float(0.0),
    } if msg is None else {
        'gps/timestamp': feature_int64(msg.header.stamp.to_nsec()),
        'gps/seq': feature_int64(msg.header.seq),
        'gps/lat': feature_float(msg.latitude),
        'gps/long': feature_float(msg.longitude),
        'gps/alt': feature_float(msg.altitude),
    }
    return gps_dict


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


class Processor(object):

    def __init__(self,
                 save_dir,
                 num_images,
                 img_format='jpg',
                 separate_streams=False,
                 debug_print=False):

        # config and helpers
        self.debug_print = debug_print
        self.separate_streams = separate_streams
        self.min_buffer_ns = 10 * SEC_PER_NANOSEC  # keep x sec of sorting/sync buffer as per image timestamps
        self.steering_offset_ns = 0  # shift steering timestamps by this much going into queue FIXME test/
        self.gps_offset_ns = 0  # shift gps timestamps by this much going into queue FIXME test
        self.bridge = CvBridge()

        # example fixed write params
        self.write_img_format = img_format
        self.write_colorspace = b'RGB'
        self.write_channels = 3

        # setup writers
        self._outdirs = []
        self._writers = []
        num_shards = num_images//4096  # at approx 40KB per image, 4K per shard gives around 160MB per shard
        if self.separate_streams:
            self._outdirs.append(get_outdir(save_dir, "left"))
            self._writers.append(ShardWriter(self._outdirs[-1], 'left', num_images, num_shards=num_shards//3))
            self._outdirs.append(get_outdir(save_dir, "center"))
            self._writers.append(ShardWriter(self._outdirs[-1], 'center', num_images, num_shards=num_shards//3))
            self._outdirs.append(get_outdir(save_dir, "right"))
            self._writers.append(ShardWriter(self._outdirs[-1], 'right', num_images, num_shards=num_shards//3))
        else:
            self._outdirs.append(get_outdir(save_dir, "combined"))
            self._writers.append(ShardWriter(self._outdirs[-1], 'combined', num_images, num_shards=num_shards))

        # stats, counts, and queues
        self.written_image_count = 0
        self.reset_queues()

    def reset_queues(self):
        self.latest_image_timestamp = None
        self._steering_queue = []   # time sorted steering heap
        self._gps_queue = []  # time sorted gps heap
        self._images_queue = []  # time sorted image heap
        self._head_steering_sample = None  # most recent steering timestamp/topic/msg sample pulled from queue
        self._head_gps_sample = None  # most recent gps timestamp/topic/msg sample pulled from queue

    def write_example(self, image_msg, steering_msg, gps_msg, dataset_id=0):
        try:
            writer = self._writers[0]
            if self.separate_streams:
                if image_msg.header.frame_id[0] == 'c':
                    writer = self._writers[1]
                elif image_msg.header.frame_id[0] == 'r':
                    writer = self._writers[2]
            image_width = 0
            image_height = 0
            if hasattr(image_msg, 'format') and 'compressed' in image_msg.format:
                buf = np.ndarray(shape=(1, len(image_msg.data)), dtype=np.uint8, buffer=image_msg.data)
                cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
                if cv_image.shape[2] != 3:
                    print("Invalid image")
                    return
                image_height = cv_image.shape[0]
                image_width = cv_image.shape[1]
                # Avoid re-encoding if we don't have to
                if check_image_format(image_msg.data) == self.write_img_format:
                    encoded = buf
                else:
                    _, encoded = cv2.imencode('.' + self.write_img_format, cv_image)
            else:
                image_width = image_msg.width
                image_height = image_msg.height
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
                _, encoded = cv2.imencode('.' + self.write_img_format, cv_image)

            feature_dict = {
                'image/timestamp': feature_int64(image_msg.header.stamp.to_nsec()),
                'image/frame_id': feature_bytes(image_msg.header.frame_id),
                'image/height': feature_int64(image_height),
                'image/width': feature_int64(image_width),
                'image/channels': feature_int64(self.write_channels),
                'image/colorspace': feature_bytes(self.write_colorspace),
                'image/format': feature_bytes(self.write_img_format),
                'image/encoded': feature_bytes(encoded.tobytes()),
                'image/dataset_id': feature_int64(dataset_id),
            }
            steering_dict = to_steering_dict(steering_msg)
            feature_dict.update(steering_dict)
            gps_dict = to_gps_dict(gps_msg)
            feature_dict.update(gps_dict)
            example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
            writer.write(example)
            self.written_image_count += 1

        except CvBridgeError as e:
            print(e)

    def push_messages(self, messages):
        for timestamp, topic, msg in messages:
            if topic in CAMERA_TOPICS:
                heapq.heappush(self._images_queue, (timestamp, topic, msg))
                if not self.latest_image_timestamp > timestamp > self.latest_image_timestamp:
                    self.latest_image_timestamp = timestamp
            elif topic == STEERING_TOPIC:
                if self.debug_print:
                    print("steering %d, %f" % (msg.header.stamp.to_nsec(), msg.steering_wheel_angle))
                timestamp += self.steering_offset_ns
                heapq.heappush(self._steering_queue, (timestamp, topic, msg))
            elif topic == GPS_FIX_TOPIC:
                if self.debug_print:
                    print("gps      %d, (%d, %d)" % (timestamp, msg.latitude, msg.longitude))
                timestamp += self.gps_offset_ns
                heapq.heappush(self._gps_queue, (timestamp, topic, msg))

    def pull_and_write(self, flush=False):
        while self.pull_ready(flush):
            image_timestamp, _, image_msg = heapq.heappop(self._images_queue)

            #FIXME implement frame filtering

            steering_samples = self._dequeue_until(self._steering_queue, image_timestamp)
            # FIXME interpolate/avg steering samples
            if steering_samples:
                self._head_steering_sample = steering_samples[-1]

            gps_samples = self._dequeue_until(self._gps_queue, image_timestamp)
            # FIXME interpolate gps samples
            if gps_samples:
                self._head_gps_sample = gps_samples[-1]

            steering_msg = self._head_steering_sample[2] if self._head_steering_sample else None
            gps_msg = self._head_gps_sample[2] if self._head_gps_sample else None
            self.write_example(image_msg, steering_msg, gps_msg)

    def pull_ready(self, flush=False):
        return self._images_queue and (flush or self._remaining_time() > self.min_buffer_ns)

    def _dequeue_until(self, queue, timestamp):
        messages = []
        while queue and queue[0][0] < timestamp:  # or <= ??
            messages.append(heapq.heappop(queue))
        return messages

    def _remaining_time(self):
        if not self._images_queue:
            return 0
        return self.latest_image_timestamp - self._images_queue[0][0]


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to tensorflow sharded records.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-b', '--indir', type=str, nargs='?', default='/data/',
        help='Input bag file')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-s', '--separate', dest='separate', action='store_true', help='Separate sets per camera')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(separate=False)
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    save_dir = args.outdir
    input_dir = args.indir
    debug_print = args.debug
    separate_streams = args.separate

    filter_topics = [STEERING_TOPIC, GPS_FIX_TOPIC] + CAMERA_TOPICS

    num_images = 0
    num_messages = 0
    bagsets = find_bagsets(input_dir, "*.bag", filter_topics)
    for bs in bagsets:
        num_images += bs.get_message_count(CAMERA_TOPICS)
        num_messages += bs.get_message_count(filter_topics)
    print("%d images, %d messages to import across %d bag sets..." % (num_images, num_messages, len(bagsets)))

    processor = Processor(
        save_dir=save_dir, num_images=num_images, img_format=img_format,
        separate_streams=separate_streams, debug_print=debug_print)

    num_read_messages = 0  # number of messages read by cursors
    for bs in bagsets:
        print("Processing set %s" % bs.name)
        sys.stdout.flush()

        cursor_group = CursorGroup(readers=bs.get_readers())
        while cursor_group:
            msg_tuples = []
            cursor_group.advance_by_until(20 * SEC_PER_NANOSEC)
            cursor_group.collect_vals(msg_tuples)
            num_read_messages += len(msg_tuples)
            processor.push_messages(msg_tuples)
            if processor.pull_ready():
                processor.pull_and_write()

        processor.pull_and_write(flush=True)  # flush remaining messages after read cursors are done
        processor.reset_queues()  # ready for next bag set

    assert num_read_messages == num_messages
    assert processor.written_image_count == num_images

    print("Completed processing %d images to TF examples." % processor.written_image_count)
    sys.stdout.flush()


if __name__ == '__main__':
    main()