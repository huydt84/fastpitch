#!/usr/bin/env bash

set -e

if [ ! -f cmudict/cmudict-0.7b ]; then
  echo "Downloading cmudict-0.7b ..."
  wget https://github.com/Alexir/CMUdict/raw/master/cmudict-0.7b -qO cmudict/cmudict-0.7b
fi
