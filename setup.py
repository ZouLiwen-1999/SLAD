from setuptools import setup, find_namespace_packages

setup(name='slad',
      packages=find_namespace_packages(include=["nnunet", "nnunet.*","CoTr", "CoTr.*","unetr", "unetr.*","segresnet","segresnet.*","resunet","resunet.*", "slad","slad.*" , 'swin_unetr', 'swin_unetr.*','nnformer', 'nnformer.*','unetr_pp','unetr_pp.*','diffunet','diffunet.*']),
      version='0.9.1',
      install_requires=[
            "tqdm==4.61.1",
            "dicom2nifti==2.3.0",
            "scikit-image==0.18.2",
            "medpy==0.4.0",
            "scipy==1.7.3",
            "batchgenerators==0.21",
            "numpy==1.21.0",
            "scikit-learn",
            "SimpleITK==2.0.2",
            "pandas==1.2.5",
            "einops==0.4.0",
            "requests==2.31.0",
            "nibabel==4.0.2",
            'tifffile==2021.11.2',
            'kornia==0.6.8',
          'timm==0.4.9',
          'fvcore==0.1.5.post20220414'
      ],
      entry_points={
          'console_scripts': [
              'nnUNet_convert_decathlon_task = nnunet.experiment_planning.nnUNet_convert_decathlon_task:main',
              'nnUNet_plan_and_preprocess = nnunet.experiment_planning.nnUNet_plan_and_preprocess:main',
              'nnUNet_train = nnunet.run.run_training:main',
              'nnUNet_train_DP = nnunet.run.run_training_DP:main',
              'nnUNet_train_DDP = nnunet.run.run_training_DDP:main',
              'nnUNet_predict = nnunet.inference.predict_simple:main',
              'nnUNet_ensemble = nnunet.inference.ensemble_predictions:main',
              'nnUNet_find_best_configuration = nnunet.evaluation.model_selection.figure_out_what_to_submit:main',
              'nnUNet_print_available_pretrained_models = nnunet.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'nnUNet_print_pretrained_model_info = nnunet.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'nnUNet_download_pretrained_model = nnunet.inference.pretrained_models.download_pretrained_model:download_by_name',
              'nnUNet_download_pretrained_model_by_url = nnunet.inference.pretrained_models.download_pretrained_model:download_by_url',
              'nnUNet_determine_postprocessing = nnunet.postprocessing.consolidate_postprocessing_simple:main',
              'nnUNet_export_model_to_zip = nnunet.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'nnUNet_install_pretrained_model_from_zip = nnunet.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'nnUNet_change_trainer_class = nnunet.inference.change_trainer:main',
              'nn_eval = nnunet.evaluation.evaluator:nnunet_evaluate_folder',
              'nnUNet_plot_task_pngs = nnunet.utilities.overlay_plots:entry_point_generate_overlay',
              'CoTr_train = CoTr.run.run_training:main',
              'CoTr_predict = CoTr.inference.predict_simple:main',
              'unetr_train = unetr.run.run_training:main',
              'unetr_predict = unetr.inference.predict_simple:main',
              'segresnet_train = segresnet.run.run_training:main',
              'segresnet_predict = segresnet.inference.predict_simple:main',
              'resunet_train = resunet.run.run_training:main',
              'resunet_predict = resunet.inference.predict_simple:main',
              'slad_train = slad.run.run_training:main',
              'slad_predict = slad.inference.predict_simple:main',
              'swin_unetr_train = swin_unetr.run.run_training:main',
              'swin_unetr_predict = swin_unetr.inference.predict_simple:main',
              'nnformer_train = nnformer.run.run_training:main',
              'nnformer_predict = nnformer.inference.predict_simple:main',
              'unetr_pp_train = unetr_pp.run.run_training:main',
              'unetr_pp_predict = unetr_pp.inference.predict_simple:main',
              'diffunet_train = diffunet.run.run_training:main',
              'diffunet_predict = diffunet.inference.predict_simple:main',
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net','CoTr','UNETR','SegResNet','ResUNet','Segment Like A Doctor','Swin UNETR','nnFormer','UNETR ++','DiffUNet']
      )
'''
0.0.1:nnunet
0.1.1:nnunet+cotr
0.1.2:排除eval命令的错误
0.2.1nnunet+cotr+unetr
0.2.2nnunet+cotr+unetr:加装einops==0.4.0包
0.3.2nnunet+cotr+unetr+segresnet
0.4.1nnunet+cotr+unetr+segresnet+resunet
0.5.1nnunet+cotr+unetr+segresnet+resunet+slad(初步建立)
0.6.1nnunet+cotr+unetr+segresnet+resunet+slad(初步建立)+swin_unetr
0.7.1nnunet+cotr+unetr+segresnet+resunet+slad(初步建立)+nnformer
0.8.1nnunet+cotr+unetr+segresnet+resunet+slad(初步建立)+nnformer+unetr_pp
0.9.1nnunet+cotr+unetr+segresnet+resunet+slad(初步建立)+nnformer+unetr_pp+diffunet
'''