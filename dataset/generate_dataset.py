import pandas as pd

# Load the datasets
df_flow = pd.read_csv('dataset/test/flow_features_test.csv')
df_basic = pd.read_csv('dataset/test/basic_features_test.csv')
df_content = pd.read_csv('dataset/test/content_features_test.csv')
df_time = pd.read_csv('dataset/test/time_features_test.csv')
df_additional = pd.read_csv('dataset/test/additional_features_test.csv')
# df_labels = pd.read_csv('dataset/test/labels_test.csv')

# Merge datasets on 'id'
merged_df = pd.merge(df_flow, df_basic, on='id', how='inner')
merged_df = pd.merge(merged_df, df_content, on='id', how='inner')
merged_df = pd.merge(merged_df, df_time, on='id', how='inner')
merged_df = pd.merge(merged_df, df_additional, on='id', how='inner')
# merged_df = pd.merge(merged_df, df_labels, on='id', how='inner')

# Reorder columns to move 'id' to the first position
columns = ['id'] + [col for col in merged_df.columns if col != 'id']
merged_df = merged_df[columns]

# Export the merged dataset
merged_df.to_csv('dataset_test.csv', index=False)
print('Successfully exported merged dataset')
