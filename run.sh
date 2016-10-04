#!/bin/bash

INPUT_DIR="/data/"
OUTPUT_DIR="/data/output"

docker run -it \
  --volume="/$(pwd)/script:/script"\
  --volume="$INPUT_DIR:/data"\
  --volume="$OUTPUT_DIR:/output"\
  "$@"
