import torch
import numpy as np
import os
import pickle

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['serif']
plt.rcParams['font.serif'] = ['Times New Roman']  + plt.rcParams['font.serif']
import numpy as np
import matplotlib
Col_show_means = True

def plot_distribution(data_to_plot, 
                    x_tick_lables=['Data 1', 'Data 2', 'Data 3'],
                    title="Distribution", xlabel="Value", ylabel="Frequency",
                    patch_artist=True, 
                    showmeans=True, 
                    meanline=True,
                    showmedians=True,
                    use_violine=False,
                    showfliers=False,
                    # boxprops=boxprops,
                    # medianprops=medianprops,
                    # flierprops=flierprops
                    ):
    """
    Plot the distribution of the data.
    """
    # # 示例数据
    # data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # data2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # data3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # # 将数据组合成一个列表
    # data_to_plot = [data1, data2, data3]

    plt.cla()
    plt.clf()
    # 创建一个图形和轴
    fig, ax = plt.subplots(figsize=(5, 3))
    # 设置标题和标签
    # ax.set_title(title)
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(y=0.0, color='b', linestyle='--')
    ax.set_facecolor('#f0f0f0')
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)

    group_num = len(data_to_plot)
    padding = 0.15
    width = (1-padding)/group_num
    names = []
    print(len(data_to_plot), len(x_tick_lables))
    for i, (data, name) in enumerate(zip(data_to_plot, x_tick_lables)):
        print(data.shape)
        # calculate group positions
        # data = data_to_plot
        data_num = len(data)
        positions = np.arange(data_num)*width + i + i%2*0. # 0.3
        if not use_violine:
            # 绘制箱线图
            boxprops = dict(linestyle='-', linewidth=1.5, color='k', facecolor='lightblue')
            medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick')
            # flierprops = dict(marker='o', markerfacecolor='k', markersize=3, linestyle='none')

            ax.boxplot(data.tolist(), 
                    patch_artist=patch_artist, 
                    showmeans=showmeans, 
                    showfliers=showfliers,
                    meanline=meanline,
                    boxprops=boxprops,
                    medianprops=medianprops,
                    positions=positions.tolist(),
                    widths=0.15,
                    )
                    # flierprops=flierprops,
                    # widths=width/data_num,

        if use_violine:
            # # 绘制小提琴图
            # violinprops = dict(linestyle='-', linewidth=3, color='k', facecolor='lightblue')
            # medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
            # flierprops = dict(marker='o', markerfacecolor='r', markersize=8, linestyle='none')

            ax.violinplot(data_to_plot, 
                    showmeans=showmeans, 
                    # meanline=meanline,
                    showmedians=showmedians,
                    # violinprops=violinprops,
                    # medianprops=medianprops,
                    # flierprops=flierprops

                    )


        # 设置x轴刻度标签
        # ax.set_xticklabels(name)
        # ax.legend()
        if data_num == 1:
            names.append(name)
        else:
            for s in range(len(data)):
                names.append(f'{name}_@{s+1}s')
    ax.set_xticklabels(names, rotation=45, fontsize=10, horizontalalignment='right')
    # 添加图例
    ax.plot([], color='firebrick', linestyle='-', linewidth=1.5, label='Median')  # 中位数图例
    ax.plot([], color='green', linestyle='--', linewidth=1.5, label='Mean')  # 均值图例
    ax.legend()
    plt.tight_layout()

    # 显示图形
    plt.savefig(f"{title}.pdf")


def plot_cllision_rate(data_to_plot,
                   x_tick_lables=['Data 1', 'Data 2', 'Data 3'],
                     title="Collision Rate", xlabel="Value", ylabel="Frequency"):
        # # 示例数据
    # data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # data2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # data3 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # # 将数据组合成一个列表
    # data_to_plot = [data1, data2, data3]

    plt.cla()
    plt.clf()
    # 创建一个图形和轴
    fig, ax = plt.subplots(figsize=(5, 3))
    # 设置标题和标签
    # ax.set_title(title)
    # ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(y=0.0, color='b', linestyle='--')
    ax.set_facecolor('#f0f0f0')
    ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.7)

    group_num = len(data_to_plot)
    padding = 0.15
    width = (1-padding)/group_num
    all_names = []
    for i, (data, name) in enumerate(zip(data_to_plot, x_tick_lables)):

        data_num = len(data)
        positions = np.arange(data_num)*width + i + i%2*0. # 0.3

        names = []
        if data_num == 1:
            names=[name]
        else:
            for s in range(len(data)):
                names.append(f'{name}_@{s+1}s')

        # print(data.max())
        # data = data > 0
        # print(data)
        # data = data.sum(axis=1) / data.shape[1]
        # data = data
        # print(data)
        # data = data.mean(axis=1)*100
        ax.bar(
            names,
            # positions.tolist(),
            data.tolist(), 
            width=0.5,
            )
            # positions=positions.tolist(),
            # widths=0.15,
            # flierprops=flierprops,
            # widths=width/data_num,


        # 设置x轴刻度标签
        # ax.set_xticklabels(name)
        # ax.legend()
        all_names.extend(names)

    ax.set_xticklabels(all_names, rotation=45, fontsize=10, horizontalalignment='right')
    # 添加图例
    # ax.plot([], color='firebrick', linestyle='-', linewidth=1.5, label='Median')  # 中位数图例
    # ax.plot([], color='green', linestyle='--', linewidth=1.5, label='Mean')  # 均值图例
    # ax.legend()
    plt.tight_layout()

    # 显示图形
    plt.savefig(f"{title}.pdf")

def collect_data_from_results(results):
    """
    Collect data from results for a specific metric.
    """
    plan_L2_1s = []
    plan_L2_2s = []
    plan_L2_3s = []
    plan_obj_col_1s = []
    plan_obj_col_2s = []
    plan_obj_col_3s = []
    plan_obj_box_col_1s = []
    plan_obj_box_col_2s = []
    plan_obj_box_col_3s = []
    for i in range(len(results)):
        plan_L2_1s.append(results[i]['metric_results']['plan_L2_1s'])
        plan_L2_2s.append(results[i]['metric_results']['plan_L2_2s'])
        plan_L2_3s.append(results[i]['metric_results']['plan_L2_3s'])
        plan_obj_col_1s.append(results[i]['metric_results']['plan_obj_col_1s'])
        plan_obj_col_2s.append(results[i]['metric_results']['plan_obj_col_2s'])
        plan_obj_col_3s.append(results[i]['metric_results']['plan_obj_col_3s'])
        plan_obj_box_col_1s.append(results[i]['metric_results']['plan_obj_box_col_1s'])
        plan_obj_box_col_2s.append(results[i]['metric_results']['plan_obj_box_col_2s'])
        plan_obj_box_col_3s.append(results[i]['metric_results']['plan_obj_box_col_3s'])
    L2 = np.array([plan_L2_1s, plan_L2_2s, plan_L2_3s])#.mean(axis=0)
    col_obj = np.array([plan_obj_col_1s, plan_obj_col_2s, plan_obj_col_3s])#.mean(axis=0)
    col_box = np.array([plan_obj_box_col_1s, plan_obj_box_col_2s, plan_obj_box_col_3s])#.mean(axis=0)
    col = col_obj + col_box
    return L2, col

if __name__ == "__main__":
    # Load the data
    # with open("/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/RollNet/outputs/250427122507_vad5/Sun_Apr_27_12_27_13_2025/pts_bbox/results_nusc.pkl", "rb") as f:
    #     results_nusc = pickle.load(f)
    # print(results_nusc.keys())
    # # print(len(results_nusc["results"]))
    # # print(results_nusc["results"]['a1bbde13230e4493a2bacce82acf9456'][0].keys())
    # dict_keys(['sample_token', 'translation', 'size', 'rotation', 'velocity', 'detection_name', 'detection_score', 'attribute_name', 'fut_traj'])

    # with open("/mnt/csi-data-aly/shared/public/xinren/DriveLM/challenge/RollNet/outputs/250427122507_vad5/Sun_Apr_27_12_27_13_2025/pts_bbox/cls_formatted.pkl", "rb") as f:
    #     cls_formatted = pickle.load(f)
    # print(cls_formatted.keys())

    baseline_path = "outputs/freezed_raw_VAD/Wed_May__7_01_26_44_2025/instance_wise_results.pkl"
    compare_path = {}
    # compare_path['vad_ind_finetune'] = "outputs/250427122507_vad5/Tue_May__6_18_05_31_2025/instance_wise_results.pkl"

    # compare_path['rollnet_tf_bev'] = "outputs/250428161735_2_6/Tue_May__6_16_34_37_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_tf_bev'] = "outputs/250428161735_2_6/rollnet_wovlm3/Tue_May__6_16_59_44_2025/instance_wise_results.pkl"


    # compare_path['rollnet_tf_bevemb'] = "outputs/250428151124_3_6/Tue_May__6_20_15_57_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_tf_bevemb'] = "outputs/250428151124_3_6/rollnet_wovlm4/Tue_May__6_16_39_48_2025/instance_wise_results.pkl"


    # compare_path['rollnet_bias_bev'] = "outputs/250506032359_2_20/Tue_May__6_15_23_09_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_bias_bev'] = "outputs/250506032359_2_20/rollnet_wovlm_2_20/Tue_May__6_16_33_57_2025/instance_wise_results.pkl"

    # compare_path['rollnet_bias_bevemb'] = "outputs/250506032409_3_20/Tue_May__6_16_20_56_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_bias_bevemb'] = "outputs/250506032409_3_20/rollnet_wovlm_3_20/Tue_May__6_16_41_09_2025/instance_wise_results.pkl"

    # freeze LM data path


    compare_path['TF_bq'] = "outputs/250512163133_c3/Mon_May_12_16_34_50_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_tf_bev'] = "outputs/250428161735_2_6/rollnet_wovlm3/Tue_May__6_16_59_44_2025/instance_wise_results.pkl"


    compare_path['TF_bf'] = "outputs/250508200218_3_25_le/Thu_May__8_20_05_16_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_tf_bevemb'] = "outputs/250428151124_3_6/rollnet_wovlm4/Tue_May__6_16_39_48_2025/instance_wise_results.pkl"


    compare_path['Bias_bq'] = "outputs/250508200205_2_25_le/Thu_May__8_20_05_14_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_bias_bev'] = "outputs/250506032359_2_20/rollnet_wovlm_2_20/Tue_May__6_16_33_57_2025/instance_wise_results.pkl"

    compare_path['Bias_bf'] = "outputs/250512172425_c4/Mon_May_12_17_28_05_2025/instance_wise_results.pkl"
    # compare_path['*rollnet_bias_bevemb'] = "outputs/250506032409_3_20/rollnet_wovlm_3_20/Tue_May__6_16_41_09_2025/instance_wise_results.pkl"
    
    with open(baseline_path, "rb") as f:
        baseline = pickle.load(f)
    bp, bc = collect_data_from_results(baseline)

    p_data_to_plot = []
    p_abs_data_to_plot = [bp]
    p_x_tick_lables = []
    p_abs_x_tick_lables = ['baseline']

    c_data_to_plot = []
    if Col_show_means:
        c_abs_data_to_plot = [bc.mean(axis=1)*100]
    else:
        c_abs_data_to_plot = [bc*100]

    c_x_tick_lables = []
    c_abs_x_tick_lables = ['baseline']

    for key in compare_path.keys():
        with open(compare_path[key], "rb") as f:
            results = pickle.load(f)
        
        p,c = collect_data_from_results(results)
        p_abs_data_to_plot.append(p)
        p_data_to_plot.append(100*(bp-p)/bp.mean(axis=1, keepdims=True))
        # p_data_to_plot.append(p)
        p_x_tick_lables.append(key)
        p_abs_x_tick_lables.append(key)
        if Col_show_means:
            c_abs_data_to_plot.append(c.mean(axis=1)*100)
            c_data_to_plot.append(100*(bc.mean(axis=1)-c.mean(axis=1))/bc.mean(axis=1))
        if not Col_show_means:
            c_abs_data_to_plot.append(c*100)
            c_data_to_plot.append(100*(bc-c)/bc.mean(axis=1, keepdims=True))
        # c_data_to_plot.append(c)
        c_x_tick_lables.append(key)
        c_abs_x_tick_lables.append(key)

    print('Load completed')
    plot_distribution(p_data_to_plot, 
                    x_tick_lables=p_x_tick_lables,
                    title="L2 Distance", xlabel="Model", ylabel="L2 Diff Reduction (%)",
                    patch_artist=True, 
                    showmeans=True, 
                    meanline=True,
                    showmedians=True,
                    use_violine=False,
                    )
    print('relative complated')
    plot_distribution(p_abs_data_to_plot, 
                    x_tick_lables=p_abs_x_tick_lables,
                    title="L2 Distance 2", xlabel="Model", ylabel="L2 Diff (m)",
                    patch_artist=True, 
                    showmeans=True, 
                    meanline=True,
                    showmedians=True,
                    use_violine=False,
                    )
    if not Col_show_means:
        print('Diff completed')
        plot_distribution(c_data_to_plot, 
                        x_tick_lables=c_x_tick_lables,
                        title="Collision Rate", xlabel="Model", ylabel="Col Reduction (%)",
                        patch_artist=True, 
                        showmeans=True, 
                        meanline=True,
                        showmedians=True,
                        use_violine=False,
                        )
        print('relative completed')
        plot_distribution(c_abs_data_to_plot, 
                        x_tick_lables=c_abs_x_tick_lables,
                        title="Collision Rate 2", xlabel="Model", ylabel="Col (m)",
                        patch_artist=True, 
                        showmeans=True, 
                        meanline=True,
                        showmedians=True,
                        use_violine=False,
                        )
        print('Col Completed')
    if Col_show_means:
        plot_cllision_rate(c_data_to_plot,
                        x_tick_lables=c_x_tick_lables,
                        title="Collision Rate", xlabel="Model", ylabel="Col Reduction (%)",
                        )
        plot_cllision_rate(c_abs_data_to_plot,
                        x_tick_lables=c_abs_x_tick_lables,
                        title="Collision Rate 2", xlabel="Model", ylabel="Col (%)",
                        )


    # plot_distribution(c_data_to_plot, 
    #                 x_tick_lables=c_x_tick_lables,
    #                 title="Collision Rate", xlabel="Metrics", ylabel="Col(%)",
    #                 patch_artist=True, 
    #                 showmeans=True, 
    #                 meanline=True,
    #                 showmedians=True,
    #                 use_violine=False,
    #                 )


    # print(len(results))
    # (Pdb) print(results[0].keys())
    # dict_keys(['pts_bbox', 'metric_results'])

    # (Pdb) print(results[0]['metric_results'].keys())
    # dict_keys(['gt_car', 'gt_pedestrian', 'cnt_ade_car', 'cnt_ade_pedestrian', 'cnt_fde_car', 'cnt_fde_pedestrian', 'hit_car', 'hit_pedestrian', 'fp_car', 'fp_pedestrian', 'ADE_car', 'ADE_pedestrian', 'FDE_car', 'FDE_pedestrian', 'MR_car', 'MR_pedestrian', 'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s', 'plan_obj_col_1s', 'plan_obj_col_2s', 'plan_obj_col_3s', 'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s', 'fut_valid_flag'])

    # (Pdb) print(results[0]['pts_bbox'].keys())      
    # dict_keys(['boxes_3d', 'scores_3d', 'labels_3d', 'trajs_3d', 'map_boxes_3d', 'map_scores_3d', 'map_labels_3d', 'map_pts_3d', 'ego_fut_preds', 'ego_fut_cmd'])

