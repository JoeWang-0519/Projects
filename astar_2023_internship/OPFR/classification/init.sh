#!/bin/sh

mkdir -p data/

conda create -n pfh-cls python=3.7 -y
conda activate pfh-cls

conda install pytorch=1.7.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch -c conda-forge -y
conda install -c anaconda h5py -y
conda install tqdm

cd modules/pointops
python3 setup.py install
cd -
