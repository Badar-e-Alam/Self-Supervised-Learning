#!/bin/bash -l
#
#SBATCH --job-name=base_model_default_recipe_1
#SBATCH --clusters=tinygpu
#SBATCH --nodes=1
# --gres and --time required to communicate the required GPU and time period
#SBATCH --gres=gpu:rtx3080:1

#SBATCH --time=24:0:0
#SBATCH --export=NONE

# useful variable
exp_name=base_model_default_recipe_1
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
cd data
echo "List of files and directories: $(ls)"
echo "Current working directory: $(pwd)"

cd $output_dir/Self-Supervised-Learning
# run training
echo "Running training ..."
python3 base_v2.py