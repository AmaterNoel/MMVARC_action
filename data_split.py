import pandas as pd

df = pd.read_excel(r'path\AR_metadata.xlsx', engine='openpyxl')
df['head_type'] = 'false'
df['middle_type'] = 'false'
df['tail_type'] = 'false'

list = [0]*140
# 提取 action_labels 列并将其拆分为整数列表
action_labels = df['action_labels']
for index in range(len(action_labels)):
    numbers = [int(num.strip()) for num in action_labels[index].split(',')]
    for num in numbers:
        list[num - 1] = list[num - 1] + 1

head_list = [index for index, num in enumerate(list) if num > 2000]
print(head_list)
middle_list = [index for index, num in enumerate(list) if 500 < num <= 2000]
print(middle_list)
tail_list = [index for index, num in enumerate(list) if num <= 500]
print(tail_list)

for index, row in df.iterrows():
    action_label = row['action_labels']
    numbers = [int(num.strip()) for num in action_label.split(',')]
    for num in numbers:
        if num - 1 in head_list:
            df.at[index, 'head_type'] = 'true'
        if num - 1 in middle_list:
            df.at[index, 'middle_type'] = 'true'
        if num - 1 in tail_list:
            df.at[index, 'tail_type'] = 'true'

df.to_excel(r'save_path\AR_metadata_eval.xlsx', engine='openpyxl')