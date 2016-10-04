#!/bin/bash

# defaults
INPUT_DIR="/data/"
OUTPUT_DIR="/data/output"
IMAGE_TAG="udacity-reader"

while getopts ":i:o:t:" opt; do
  case $opt in
    i) INPUT_DIR=$OPTARG ;;
    o) OUTPUT_DIR=$OPTARG ;;
    t) IMAGE_TAG=$OPTARG ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done
shift $(expr $OPTIND - 1)

#POS1=${@:$OPTIND:1}

echo "Running bagdump with input dir '$INPUT_DIR', output dir '$OUTPUT_DIR', docker image '$IMAGE_TAG'..."

docker run --rm -it \
  --volume="/$(pwd)/script:/script"\
  --volume="$INPUT_DIR:/data"\
  --volume="$OUTPUT_DIR:/output"\
  $IMAGE_TAG\
  "$@"
