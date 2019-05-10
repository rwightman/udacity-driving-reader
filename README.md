# udacity-driving-reader

Scripts to read and dump data from the rosbag format used in the udacity self-driving dataset(s).

The scripts were setup to run within a Docker container so that I can extract the data from the rosbag format without needing to install ROS on my system. The docker container is built from the ROS kinetic-perception image. I've added python modules and the latest Tensorflow on top of that.

I've run this code on Ubuntu 16.04 and 18.04 with Docker CE installed as per https://docs.docker.com/install/linux/docker-ce/ubuntu/. No other platform has been tried.

Since the original release of this script, this latest iteration has been updated to support bag files with compressed images and bag files that have been split into multiple files by time or topics. With support for this, support for a reordering buffer was added to bag2tf. 

The latest versions scan all bag files and extract their info in yaml format before doing a second pass to read the data, this adds some time but provides a mechanism for supporting the variety of bag formats and splits now being used in the datasets. The info yaml files are also dumped as part of the bagdump process.

## Installation

Checkout this code and run in place. I have not pushed docker container to hub.

## Usage


Build the docker container manually or using ./build.sh before executing any of the run scripts.

Run one of the run scripts for dumping to images + csv or Tensorflow sharded records files.

This and future versions of the scripts expect all datasets to exist in SEPARATE folders with only bag files for the same dataset in each folder. The input folder should thus be a folder with one folder per dataset. The bagdump script will mirror those input folders in the output, while the bag2tf will combine them all into one sharded stream.

The paths passed to the run scripts are used as docker volume mappings. **These paths must be absolute paths on your local filesystem (relative to the root)**. Keep this in mind if you try to change the input/output args.

### Dump to images + CSV

    ./run-bagdump.sh -i [absolute dir with folders containing bag files] -o [absolute output dir] -- [args to pass to python script]

For example, if your dataset bags are in /data/dataset2-1/dataset.bag, /data/udacity-datasetElCamino/*.bag etc., and you'd like the output in /output:

    ./run-bagdump.sh -i /data -o /output

The same as above, but you want to convert to png instead of jpg:

    ./run-bagdump.sh -i /data -o /output -- -f png

### Dump to Tensorflow sharded files

Same basic arguments as for bagdump above. There are some additional arguments of note to pass to the python script.

The default arguments write all cameras into the same sharded stream along with latest steering entry. To write images to three separate streams, one for each camera, add an -s (or --separate) argument.

i.e.

    ./run-bag2tf.sh -i /data -o /output -- --separate

