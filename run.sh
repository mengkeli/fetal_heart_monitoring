#!/bin/bash

rm -rf data/data.csv
rm -rf data/info.csv

cd code
python import.py
python svm.py