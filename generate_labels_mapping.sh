#!/bin/bash
python generate_labels_mapping.py
if [ $? != 0 ];
then
    echo "Failed to generate labels mapping"
else
	echo "Generated labels mapping"
fi