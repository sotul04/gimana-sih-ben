import pandas as pd
df_additional = pd.read_csv('dataset/train/additional_features_train.csv')
df_basic = pd.read_csv('dataset/train/basic_features_train.csv')
df_content = pd.read_csv('dataset/train/content_features_train.csv')
df_flow = pd.read_csv('dataset/train/flow_features_train.csv')
df_labels = pd.read_csv('dataset/train/labels_train.csv')
df_time = pd.read_csv('dataset/train/time_features_train.csv')

merged_df = pd.merge(df_additional, df_basic, on='id', how='inner')
merged_df = pd.merge(merged_df, df_content, on='id', how='inner')
merged_df = pd.merge(merged_df, df_flow, on='id', how='inner')
merged_df = pd.merge(merged_df, df_labels, on='id', how='inner')
merged_df = pd.merge(merged_df, df_time, on='id', how='inner')

merged_df.to_csv('dataset_train.csv', index=False)
print('Successfully export merged dataset')