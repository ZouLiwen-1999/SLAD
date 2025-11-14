# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------

import argparse
from batchgenerators.utilities.file_and_folder_operations import *
from slad.run.default_configuration import get_default_configuration
from slad.paths import default_plans_identifier
from nnunet.training.cascade_stuff.predict_next_stage import predict_next_stage, predict_next_stage_fast
from slad.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import os
import slad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu", type=str, default='0')
    parser.add_argument("network", type=str, default='3d_lowres') 
    parser.add_argument("network_trainer", type=str, default='SladTrainerV4_Stage1_V1')
    parser.add_argument("task", type=str, default='083', help="can be task name or task id")
    parser.add_argument("fold", type=str, default='0', help='0, 1, ..., 5 or \'all\'')
    
#     parser.add_argument("-amae", '--use_amae', default=False, help="use this if you want to run the Anatomy-aware Masked AutoEncoder(AMAE) module",
#                         required=False, action="store_true")
#     parser.add_argument("-cgrm", '--use_cgrm' ,default=False, help="use this if you want to run the Casuality-driven Graph Reasoning Module (CGRM)",
#                         required=False, action="store_true")
#     parser.add_argument("-dfdm", '--use_dfdm', default=False, help="use this if you want to run the Distraction-sensitive Feature Distillation Module (DFDM)", required=False, action="store_true")
    parser.add_argument("-val", "--validation_only", default=False, help="use this if you want to only run the validation",
                        required=False, action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("-p", help="plans identifier. Only change this if you created a custom experiment planner",
                        default=default_plans_identifier, required=False)
    
    
    
    parser.add_argument("--use_compressed_data", default=False, action="store_true",
                        help="If you set use_compressed_data, the training cases will not be decompressed. Reading compressed data "
                             "is much more CPU and RAM intensive and should only be used if you know what you are "
                             "doing", required=False)
    parser.add_argument("--deterministic", default=False, action="store_true")
    parser.add_argument("--npz", required=False, default=False, action="store_true", help="if set then nnUNet will "
                                                                                          "export npz files of "
                                                                                          "predicted segmentations "
                                                                                          "in the validation as well. "
                                                                                          "This is needed to run the "
                                                                                          "ensembling step so unless "
                                                                                          "you are developing nnUNet "
                                                                                          "you should enable this")
    parser.add_argument("--find_lr", required=False, default=False, action="store_true",
                        help="not used here, just for fun")
    
    parser.add_argument("--valbest", default=True, help="hands off. This is not intended to be used")#zlw默认改为--valbest=True
    
    parser.add_argument("--fp32", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")
    
    parser.add_argument("--pred_next_stage_fast", required=False, default=False, action="store_true",
                        help="disable mixed precision training and run old school fp32")#zlw
    
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--disable_saving", required=False, action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    task = args.task
    fold = args.fold
    network = args.network
    network_trainer = args.network_trainer
    validation_only = args.validation_only

    plans_identifier = args.p
    find_lr = args.find_lr
    
#     #zlw et al. Segment Like A Doctor
#     use_amae = args.use_amae
#     use_cgrm = args.use_cgrm
#     use_dfdm = args.use_dfdm
#     outpath = 'amae'+ str(int(use_amae)) + '_'+ 'cgrm'+ str(int(use_cgrm)) + '_' + 'dfdm'+ str(int(use_dfdm))
    
    use_compressed_data = args.use_compressed_data
    decompress_data = not use_compressed_data

    deterministic = args.deterministic
    valbest = args.valbest
#     print('valbest:',valbest)

    fp32 = args.fp32
    pred_next_stage_fast=args.pred_next_stage_fast
    run_mixed_precision = not fp32

    val_folder = args.val_folder

    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)

    if fold == 'all':
        pass
    else:
        fold = int(fold)
    print('=============================================Let it Segment Like A Doctor=============================================')
#     print('')
#     plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
#     trainer_class = get_default_configuration(outpath, network, task, network_trainer, plans_identifier, \
#                                               search_in=(slad.__path__[0], "training", "network_training"), \
#                                               base_module='slad.training.network_training')
    
    plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
    trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier, \
                                              search_in=(slad.__path__[0], "training", "network_training"), \
                                              base_module='slad.training.network_training')
    
    print('plans_file:',plans_file)
    print('output_folder_name:',output_folder_name)
    print('dataset_directory:',dataset_directory)
    print('batch_dice:',batch_dice)
    print('stage:',stage)
    print('trainer_class:',trainer_class)
    
    
    trainer = trainer_class(plans_file, fold, output_folder=output_folder_name, dataset_directory=dataset_directory,
                            batch_dice=batch_dice, stage=stage, unpack_data=decompress_data,
                            deterministic=deterministic,
                            fp16=run_mixed_precision)
    
    
    
    if args.disable_saving:
        trainer.save_latest_only = False  # if false it will not store/overwrite _latest but separate files each
        trainer.save_intermediate_checkpoints = False  # whether or not to save checkpoint_latest
        trainer.save_best_checkpoint = False  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        trainer.save_final_checkpoint = False  # whether or not to save the final checkpoint
    
    print('Initialize the network...')
    trainer.initialize(not validation_only) #网络初始化

    

    if find_lr:
        trainer.find_lr()
        
    else: #大部分直接走这步
        if not validation_only: #如果训练的话
#             print('start to train......')
            if args.continue_training: #如果继续训练
                trainer.load_latest_checkpoint()
            trainer.run_training() #主要功能就在这个run_training()
            
#             return
        
        else: #如果只做验证的话
            if valbest:
                trainer.load_best_checkpoint(train=False)
            else:
                trainer.load_latest_checkpoint(train=False)
        
#         return
    
        trainer.network.eval() #改为eval()模式

        # predict validation
        trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder) #验证一下

        if network == '3d_lowres':
            if not pred_next_stage_fast:
                print("predicting segmentations for the next stage of the cascade")
                predict_next_stage(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))
            else: #zlw
                print("predicting segmentations for the next stage of the cascade with the fast mode")
                predict_next_stage_fast(trainer, join(dataset_directory, trainer.plans['data_identifier'] + "_stage%d" % 1))


if __name__ == "__main__":
    main()
