#!/usr/bin/env bash

set -e

if [ ! -f cmudict/cmudict-0.7b ]; then
  echo "Downloading cmudict-0.7b ..."
  wget https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b -qO cmudict/cmudict-0.7b
fi

DATA_DIR="LJSpeech-1.1"
LJS_ARCH="LJSpeech-1.1.tar.bz2"
LJS_URL="http://data.keithito.com/data/speech/${LJS_ARCH}"

if [ ! -d ${DATA_DIR} ]; then
  echo "Downloading ${LJS_ARCH} ..."
  wget ${LJS_URL}
  echo "Extracting ${LJS_ARCH} ..."
  tar jxvf ${LJS_ARCH}
  rm -f ${LJS_ARCH}
fi
