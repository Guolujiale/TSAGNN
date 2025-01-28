import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('data/20-21data.csv')

# 加载business_id这一列
business_ids = df['business_id']

# 统计不同business_id的种类
unique_business_ids = business_ids.nunique()

# 输出结果
print(f'共有 {unique_business_ids} 种不同的 business_id')

# 使用groupby统计每个business_id的样本数量
business_id_counts = df.groupby('business_id').size()

# 打印所有不同的样本数量
unique_counts = business_id_counts.value_counts().index.tolist()
print("\n所有不同的样本数量：")
print(unique_counts)

# 检查所有类别的样本数量是否相同
if business_id_counts.nunique() == 1:
    print("\n所有类别中的样本数量相同")
else:
    print("\n类别中的样本数量不同")

# 使用factorize将business_id转化为从0开始的数字
df['business_id_encoded'], _ = pd.factorize(df['business_id'])

# 对total_food_positive, total_service_positive, total_waiting_positive, total_ambiance_positive的值除以text_count
df['total_food_positive'] /= df['text_count']
df['total_service_positive'] /= df['text_count']
df['total_waiting_positive'] /= df['text_count']
df['total_ambiance_positive'] /= df['text_count']

# 对state这一列进行编码，从0开始，相同的值使用相同的编码
df['state_encoded'], _ = pd.factorize(df['state'])

# 对以下列以及state_encoded进行ln归一化
columns_to_normalize = ['total_food_positive', 'total_service_positive', 'total_waiting_positive', 
                        'total_ambiance_positive', 'avg_text_length', 'state_encoded']
df[columns_to_normalize] = np.log1p(df[columns_to_normalize])

# 对average_monthly_stars进行四舍五入，保留小数点后一位
df['average_monthly_stars'] = df['average_monthly_stars'].round(1)

# 将average_monthly_stars和placekey列移动到最后
columns = [col for col in df.columns if col not in ['average_monthly_stars', 'placekey']]
columns.extend(['average_monthly_stars', 'placekey'])
df = df[columns]

# 将结果保存到新的CSV文件
df.to_csv('data/20-21data_encoded.csv', index=False)

# 输出确认信息
print("已将编码后的数据保存到 'data/20-21data_encoded.csv'")
