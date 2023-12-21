#!/bin/bash -l
#
#SBATCH --job-name=multiclass
#SBATCH --clusters=tinygpu
#SBATCH --nodes=1
# --gres and --time required to communicate the required GPU and time period
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:0:0
#SBATCH --export=NONE

# useful variable
exp_name=multiclass
output_dir=/home/woody/rzku/mrvl005h/output/$exp_name
scratch_dir=/scratch/mrvl005h

# load modules
echo "Loading modules ..."
source /home/hpc/rzku/mrvl005h/miniconda3/bin/activate
conda activate ffcv
module load cuda

# copy the code to the output dir
echo "Preparing code dir ..."
mkdir $output_dir
cp -r /home/vault/rzku/mrvl005h/Self-Supervised-Learning $output_dir
#cd $output_dir/lightly/examples/pytorch/

# prepare dataset
echo "Preparing datasets ..."
mkdir $scratch_dir
cd $scratch_dir

echo" copying data to scratch ..."
mkdir $$exp_name
cd $exp_name
echo "copying code to scratch ...:pwd"
echo "unzipping data ..."
cp /home/vault/rzku/mrvl005h/data/Image_data.zip .
echo "Current working directory: $(pwd)"
unzip Image_data.zip
cd Image_data
echo "List of files and directories: $(ls)"
echo "Current working directory: $(pwd)"

cd $output_dir/Self-Supervised-Learning
# run training
echo "Running training ..."
python3 classifier.py