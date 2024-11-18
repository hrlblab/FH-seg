import os
import pandas as pd


#root_dir = '/data2/DoDNet/MIDL/Val_for_Mice_Universeg_10class/Validation_normal'
#root_dir = '/data2/DoDNet/MIDL/Val_for_Hum_Omniseg_10class/Validation_normal'
#root_dir = '/data2/DoDNet/MIDL/Mice2_validation_scale_normalwhole_1217'
#root_dir = '/data2/DoDNet/MIDL/Test_for_Hum_Omniseg_10class/Validation_normal'
#root_dir = '/data2/DoDNet/MIDL/Hum_validation_scale_normalwhole_1217'
#root_dir = '/data2/DoDNet/MIDL/Test_for_H2H_Omniseg_10class/Validation_normal'
#root_dir = '/data2/DoDNet/MIDL/DeepLabV3_for_10_class_human/Human_validation'
#root_dir = '/data2/DoDNet/MIDL/VL_H_TR_M&H_for_10class_fix_Omniseg/HOnly_validation'
#root_dir = '/data2/DoDNet/Universeg_mice_10class_TM&VH/Mice_validation'
root_dir = '/data2/DoDNet/MIDL/Test_for_MH2H_Omniseg_10class/Validation_normal'
#root_dir = '/data2/DoDNet/Segmenter_10class_human/HC_validation'
final_results = pd.DataFrame(columns=['Subfolder', 'Dice Mean'])


for subdir, dirs, files in os.walk(root_dir):
    for file in files:

        #if file == 'validation_result.csv':
        if file == 'testing_result.csv':

            file_path = os.path.join(subdir, file)

            df = pd.read_csv(file_path)
            #selected_rows = df.iloc[[2, 4, 5, 6, 7, 8, 9]]
            selected_rows = df.iloc[[2]]
            #selected_rows = df.iloc[[2, 4, 5, 6, 7, 8, 9]]
            dice_mean = selected_rows['Dice'].mean()

            subfolder_name = os.path.basename(subdir)

            final_results = final_results.append({'Subfolder': subfolder_name, 'Dice Mean': dice_mean}, ignore_index=True)
            final_results.sort_values(by='Dice Mean', ascending=False, inplace=True)

final_results.to_csv('sub_final.csv', index=False)
