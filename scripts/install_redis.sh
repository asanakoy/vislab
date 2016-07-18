#!/bin/bash -x
wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make -j 4

cp src/redis-server ~/.local/bin/
cp src/redis-cli ~/.local/bin/