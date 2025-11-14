import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "SLAD"#model/下框架的名字
default_plans_identifier = "nnUNetPlansv2.1"
default_data_identifier = 'nnUNetData_plans_v2.1'
default_trainer = "SladTrainerV4_Stage1_V1"
default_cascade_trainer = "SladTrainerV4_Stage2_V1"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

base='/opt/data/private/work/SLAD/v4/data'
preprocessing_output_dir ='/opt/data/private/work/SLAD/v4/data/preprocess'
network_training_output_dir_base='/opt/data/private/work/SLAD/v4/data/result'
# network_training_output_dir_base='/opt/data/private/work/SLAD/v4/data/mia_minor' #为了MIA的小修临时更改结果文件夹

if base is not None:
    nnUNet_raw_data = join(base, "raw")
    nnUNet_cropped_data = join(base, "cropped")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
else:
    print("nnUNet_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    nnUNet_cropped_data = nnUNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information on how to set this "
          "up.")
    network_training_output_dir = None
