# Copyright (C) 2016 Ross Wightman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
import os
import cv2
import rosbag
import argparse
import pandas as pd

LEFT_CAMERA_TOPIC = "/left_camera/image_color"
CENTER_CMAERA_TOPIC = "/center_camera/image_color"
RIGHT_CAMERA_TOPIC = "/right_camera/image_color"
STEERING_TOPIC = "/vehicle/steering_report"


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def write_image(bridge, outdir, msg, fmt='png', table=None):
    image_name = os.path.join(outdir, str(msg.header.stamp.to_nsec()) + '.' + fmt)
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(image_name, cv_image)
        if table is not None:
            table["seq"].append(msg.header.seq)
            table["timestamp"].append(msg.header.stamp.to_nsec())
            table["width"].append(msg.width)
            table["height"].append(msg.height)
            table["frame_id"].append(msg.header.frame_id)
            table["filename"].append(image_name)      
    except CvBridgeError as e:
        print(e)
    return image_name


def main():
    parser = argparse.ArgumentParser(description='Convert rosbag to images and csv.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/output',
        help='Output folder')
    parser.add_argument('-b', '--bagfile', type=str, nargs='?', default='/data/dataset.bag',
        help='Input bag file')
    parser.add_argument('-f', '--img_format', type=str, nargs='?', default='jpg',
        help='Image encode format, png or jpg')
    parser.add_argument('-d', dest='debug', action='store_true', help='Debug print enable')
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    img_format = args.img_format
    save_dir = args.outdir
    rosbag_file = args.bagfile
    debug_print = args.debug

    bridge = CvBridge()

    include_images = True
    if include_images:
        filter_topics = [LEFT_CAMERA_TOPIC, CENTER_CMAERA_TOPIC, RIGHT_CAMERA_TOPIC, STEERING_TOPIC]
        left_outdir = get_outdir(save_dir, "left")
        center_outdir = get_outdir(save_dir, "center")
        right_outdir = get_outdir(save_dir, "right")
    else:
        filter_topics = [STEERING_TOPIC]

    camera_cols = ["seq", "timestamp", "width", "height", "frame_id", "filename"]
    camera_dict = defaultdict(list)

    steering_cols = ["seq", "timestamp", "angle", "torque", "speed"]
    steering_dict = defaultdict(list)

    with rosbag.Bag(rosbag_file, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=filter_topics):
            if topic == LEFT_CAMERA_TOPIC:
                if debug_print:
                    print 'l_camera ' + str(msg.header.stamp.to_nsec())
                write_image(bridge, left_outdir, msg, fmt=img_format, table=camera_dict)
            elif topic == CENTER_CMAERA_TOPIC:
                if debug_print:
                    print 'c_camera ' + str(msg.header.stamp.to_nsec())
                write_image(bridge, center_outdir, msg, fmt=img_format, table=camera_dict)
            elif topic == RIGHT_CAMERA_TOPIC:
                if debug_print:
                    print 'r_camera ' + str(msg.header.stamp.to_nsec())
                write_image(bridge, right_outdir, msg, fmt=img_format, table=camera_dict)
            elif topic == STEERING_TOPIC:
                if debug_print:
                    print 'steering %u : %f, %f' % (msg.header.stamp.to_nsec(), msg.steering_wheel_angle)
                steering_dict["seq"].append(msg.header.seq)
                steering_dict["timestamp"].append(msg.header.stamp.to_nsec())
                steering_dict["angle"].append(msg.steering_wheel_angle)
                steering_dict["torque"].append(msg.steering_wheel_torque)
                steering_dict["speed"].append(msg.speed)

    camera_csv_path = os.path.join(save_dir, 'camera.csv')
    camera_df = pd.DataFrame(data=camera_dict, columns=camera_cols)
    camera_df.to_csv(camera_csv_path, index=False)

    steering_csv_path = os.path.join(save_dir, 'steering.csv')
    steering_df = pd.DataFrame(data=steering_dict, columns=steering_cols)
    steering_df.to_csv(steering_csv_path, index=False)

if __name__ == '__main__':
    main()