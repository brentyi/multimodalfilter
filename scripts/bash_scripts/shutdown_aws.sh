#!/bin/bash
while [ 1 ]
do
    out=`ps -U ubuntu | grep python`
    if [ -z "$out" ]
    then
        shutdown
    else
        echo "waiting"
    fi
    sleep 5
done
