# folder_name=$1
# folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/test/rollnet_base_vis_2_25_le/Mon_May_19_17_05_14_2025
# folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/test/rollnet_base_vis_2_25_le/Tue_May_20_02_38_32_2025_ori
# folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/test/VAD_base_stage_2/Sun_May_18_00_15_23_2025
# folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/outputs/freezed_raw_VAD/Wed_May__7_01_26_44_2025
# folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/test/rollnet_base_vis_3_26/Tue_May_20_01_16_46_2025_ori
# folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/test/rollnet_base_vis_2_25_le/Fri_Aug__8_18_01_04_2025
folder_name=/media/rxin/Rex/DriveLM/challenge/RollNet/test/rollnet_base_vis_3_26/Sat_Aug__9_10_30_53_2025
python tools/analysis_tools/visualization.py \
--result-path $folder_name/pts_bbox/results_nusc.pkl \
--save-path $folder_name/pts_bbox/plots
