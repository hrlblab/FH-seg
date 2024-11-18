import os
import pandas as pd


#root_dir = '/data2/DoDNet/MIDL/Val_for_Hum_Omniseg_10class/Validation_normal'

root_dir = '/data2/DoDNet/Segmenter_10class_human/HC_validation'


final_results = pd.DataFrame(columns=['Subfolder', 'Dice Mean'])


for subdir, dirs, files in os.walk(root_dir):
    for file in files:

        if file == 'validation_result.csv':
        #if file == 'testing_result.csv':
            file_path = os.path.join(subdir, file)

            df = pd.read_csv(file_path)

            dice_mean = df['Dice'].mean()

            subfolder_name = os.path.basename(subdir)

            final_results = final_results.append({'Subfolder': subfolder_name, 'Dice Mean': dice_mean}, ignore_index=True)
            final_results.sort_values(by='Dice Mean', ascending=False, inplace=True)

final_results.to_csv('final_mice.csv', index=False)
