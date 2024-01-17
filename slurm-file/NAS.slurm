#!/bin/bash -l
#
#SBATCH --job-name=NAS_exp_small
#SBATCH --clusters=tinygpu
#SBATCH --nodes=1
# --gres and --time required to communicate the required GPU and time period
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=24:0:0
#SBATCH --export=NONE

# useful variable

exp_name=NAS_exp_small
output_dir=/home/woody/rzku/mrvl005h/output/$exp_name
#scratch_dir=/scratch/mrvl005h

# load modules
echo "Loading modules ..."
source /home/hpc/rzku/mrvl005h/miniconda3/bin/activate
conda activate ffcv
module load cuda

# copy the code to the output dir
echo "Preparing code dir ..."
mkdir $output_dir
cp -r /home/vault/rzku/mrvl005h/Neural-Architecture-Search $output_dir
cd $output_dir//Neural-Architecture-Search/classification/

# prepare dataset
echo "Preparing datasets ..."
cp -r /home/vault/rzku/mrvl005h/.medmnist $HOME

# run training
echo "Running training ..."
python3 use_case.py