import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from collections import defaultdict
import random

def generate_edges(input_data):
    """
    生成边的函数：以序列的相邻关系构建边
    """
    n = len(input_data)
    edges = defaultdict(list)

    for i in range(n - 1):
        edges['input'].append((i, i + 1))

    return edges

def load_data(filename):
    """
    数据加载函数
    """
    df = pd.read_csv(filename, encoding='ISO-8859-1')
    return df

class HeteroGraphDataset(InMemoryDataset):
    def __init__(self, 
                 filename, 
                 win_size, 
                 transform=None, 
                 output_dir="./data/graph_data"):
        self.root = '.'  # 父类InMemoryDataset要求定义
        self.win_size = win_size
        self.output_dir = output_dir
        super(HeteroGraphDataset, self).__init__(self.root, transform)

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 直接处理数据，不再检测是否已经存在
        df = load_data(filename)
        self.data, self.edge_index_list = self.process_data(df)

    def process_data(self, df):
        """
        针对整个数据集进行处理：生成样本数据、边数据、以及每个样本的标签
        """
        feature_name = df.iloc[:, 0].values
        input_feature = df.iloc[:, 7:13].values  # 7~12 共6维特征
        labels = df.iloc[:, 16].values
        business_ids = df['business_id_encoded'].values

        # 用于在返回时存储所有样本（Graph对象）以及所有edge_index
        data_list = []
        edge_index_list_all = []
        sample_info_list = []  # 用于保存样本信息 (business_id, sample_index, Data, edge_index_list)

        # 按照业务ID进行分类
        unique_business_ids = np.unique(business_ids)

        for business_id in unique_business_ids:
            business_df = df[df['business_id_encoded'] == business_id]
            business_feature_name = business_df.iloc[:, 0].values
            business_input_feature = business_df.iloc[:, 7:13].values
            business_labels = business_df.iloc[:, 16].values

            num_points = len(business_input_feature)
            num_samples = num_points - self.win_size + 1
            if num_samples <= 0:
                continue

            for i in range(num_samples):
                start_idx = i
                end_idx = i + self.win_size

                sample_feature_name = business_feature_name[start_idx:end_idx]
                sample_input_feature = business_input_feature[start_idx:end_idx]
                sample_labels = business_labels[start_idx:end_idx]

                input_feature_values = torch.tensor(sample_input_feature, dtype=torch.float)
                label_value = torch.tensor(sample_labels[-1], dtype=torch.float)

                # 生成边
                edges = generate_edges(sample_input_feature)
                num_nodes = len(sample_input_feature)
                data = Data(num_nodes=num_nodes)

                edge_index_list = []
                for edge_type, edge_list in edges.items():
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    edge_index_list.append(edge_index)
                    setattr(data, f'{edge_type}_index', edge_index)

                data.star = label_value
                data.input = input_feature_values

                sample_info_list.append((business_id, i + 1, data, edge_index_list))

        # 4) 保存所有数据为一个pt文件
        self.save_samples(sample_info_list)

        # 为了后续能够通过 __getitem__ 进行索引，构建 data_list 和 edge_index_list_all
        for (business_id, sample_idx, data, edge_index_list) in sample_info_list:
            data_list.append(data)
            edge_index_list_all.append(edge_index_list)

        return data_list, edge_index_list_all

    def save_samples(self, sample_info_list):
        """
        将所有样本保存为一个pt文件和一个csv文件，并按业务ID划分训练集和测试集
        """
        # 用来收集样本，按 business_id 分组
        bizid2samples = defaultdict(list)

        for (business_id, sample_idx, data, edge_index_list) in sample_info_list:
            # 保存所有数据
            bizid2samples[business_id].append((sample_idx, data, edge_index_list))

        # 划分训练集和测试集
        train_samples = []
        test_samples = []

        # 1/5划分规则：每个business_id下随机选取1/5作为测试集，剩下的为训练集
        for business_id, samples in bizid2samples.items():
            random.shuffle(samples)
            num_test = len(samples) // 5
            test_samples_biz = samples[:num_test]
            train_samples_biz = samples[num_test:]

            train_samples.extend(train_samples_biz)
            test_samples.extend(test_samples_biz)

        # 保存训练集和测试集
        self.save_data_to_file(train_samples, 'train')
        self.save_data_to_file(test_samples, 'test')

    def save_data_to_file(self, samples, data_type):
        """
        保存样本数据到 .pt 和 .csv 文件
        """
        all_data = []
        all_edge_indices = []
        sample_info_list = []

        for sample_idx, data, edge_index_list in samples:
            all_data.append({
                'input': data.input,
                'star': data.star
            })
            all_edge_indices.append(edge_index_list)

            # 记录样本信息
            for edge_index in edge_index_list:
                edge_data = edge_index.cpu().numpy().tolist()
                sample_info_list.append({
                    'sample_idx': sample_idx,
                    'input_features': data.input.tolist(),
                    'star': data.star.item(),
                    'edge_index': edge_data
                })

        # 保存为 .pt 文件
        torch.save({
            'data': all_data,
            'edge_indices': all_edge_indices
        }, os.path.join(self.output_dir, f"{data_type}_data.pt"))

        # 保存为 .csv 文件
        df = pd.DataFrame(sample_info_list)
        df.to_csv(os.path.join(self.output_dir, f"{data_type}_data.csv"), index=False)

    def len(self):
        """
        返回数据集长度，InMemoryDataset 要求实现
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回 idx 对应的 (Data, 对应EdgeIndexList)
        """
        return self.data[idx], self.edge_index_list[idx]
'''
# 使用示例
filename = "data/20-21data_encoded.csv"
win_size = 3  # 窗口大小为3
dataset = HeteroGraphDataset(filename, win_size, output_dir="./data/graph_data")
data, edge_index_list = dataset[0]
print(data)
print(edge_index_list)
'''