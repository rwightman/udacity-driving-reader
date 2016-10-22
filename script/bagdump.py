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
from collections import defaultdict
import os
import sys
import glob
import cv2
import rosbag
import argparse
import pandas as pd

LEFT_CAMERA_TOPIC = "/left_camera/image_color"
CENTER_CAMERA_TOPIC = "/center_camera/image_color"
RIGHT_CAMERA_TOPIC = "/right_camera/image_color"
CAMERA_TOPICS = [LEFT_CAMERA_TOPIC, CENTER_CAMERA_TOPIC, RIGHT_CAMERA_TOPIC]
STEERING_TOPIC = "/vehicle/steering_report"


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def uncompress_image(compressed_msg, encoding):
    """
    Take a sensor_msgs/CompressedImage and encoding
    This will assume the compression has ignored the encoding and
    will apply the encoding
    return a sensor_msgs/Image
    """
    fh = BytesIO(compressed_msg.data)
    img = Image.open(fh)
 
    output_msg = smImage()
    output_msg.header = compressed_msg.header
    output_msg.width, output_msg.height = img.size
    output_msg.encoding = encoding
    output_msg.is_bigendian = False  # TODO
    output_msg.step = output_msg.width
    output_msg.data = img.tostring()
    return output_msg

def write_image(bridge, outdir, msg, fmt='png'):
    image_filename = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(image_filename, cv_image)
    except CvBridgeError as e:
        print(e)
    return image_filename


def main():
    compressed_images=True
    
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-b', '--bagfiles', type=str, nargs='?', default='/data/*.bag',
        help='Input bag file or pattern')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    base_outdir = args.outdir
    rosbag_pattern = args.bagfiles
    debug_print = args.debug

    bridge = CvBridge()

    include_images = True
    if include_images:
        filter_topics = [LEFT_CAMERA_TOPIC, CENTER_CAMERA_TOPIC, RIGHT_CAMERA_TOPIC, STEERING_TOPIC]
    else:
        filter_topics = [STEERING_TOPIC]

    bagfiles = glob.glob(rosbag_pattern)
    if not bagfiles:
        print("No bagfiles found matching %s" % rosbag_pattern)
        exit(1)

    for bagfile in bagfiles:
        print("Processing bag %s" % bagfile)
        sys.stdout.flush()

        dataset_name = os.path.splitext(os.path.basename(bagfile))[0]
        dataset_dir = get_outdir(base_outdir, dataset_name)
        left_outdir = get_outdir(dataset_dir, "left")
        center_outdir = get_outdir(dataset_dir, "center")
        right_outdir = get_outdir(dataset_dir, "right")

        camera_cols = ["seq", "timestamp", "width", "height", "frame_id", "filename"]
        camera_dict = defaultdict(list)

        steering_cols = ["seq", "timestamp", "angle", "torque", "speed"]
        steering_dict = defaultdict(list)

        with rosbag.Bag(bagfile, "r") as bag:
            for topic, msg, t in bag.read_messages(topics=filter_topics):
                if topic in CAMERA_TOPICS:
                    if topic[1] == 'l':
                        outdir = left_outdir
                    elif topic[1] == 'c':
                        outdir = center_outdir
                    elif topic[1]  == 'r':
                        outdir = right_outdir
                    if debug_print:
                        print("%s_camera %d" % (topic[1], msg.header.stamp.to_nsec()))

                    if compressed_images:
                        msg = uncompress_image(msg, "bgr8")
                        image_filename = write_image(bridge, outdir, msg, fmt=img_format)
                    else:
                        image_filename = write_image(bridge, outdir, msg, fmt=img_format)
                    camera_dict["seq"].append(msg.header.seq)
                    camera_dict["timestamp"].append(msg.header.stamp.to_nsec())
                    camera_dict["width"].append(msg.width)
                    camera_dict["height"].append(msg.height)
                    camera_dict["frame_id"].append(msg.header.frame_id)
                    camera_dict["filename"].append(os.path.relpath(image_filename, dataset_dir))

                elif topic == STEERING_TOPIC:
                    if debug_print:
                        print("steering %d %f" % (msg.header.stamp.to_nsec(), msg.steering_wheel_angle))

                    steering_dict["seq"].append(msg.header.seq)
                    steering_dict["timestamp"].append(msg.header.stamp.to_nsec())
                    steering_dict["angle"].append(msg.steering_wheel_angle)
                    steering_dict["torque"].append(msg.steering_wheel_torque)
                    steering_dict["speed"].append(msg.speed)

        camera_csv_path = os.path.join(dataset_dir, 'camera.csv')
        camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
        camera_df.to_csv(camera_csv_path, index=False)

        steering_csv_path = os.path.join(dataset_dir, 'steering.csv')
        steering_df = pd.DataFrame(data=steering_dict, columns=steering_cols)
        steering_df.to_csv(steering_csv_path, index=False)

if __name__ == '__main__':
    main()
