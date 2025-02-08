import torch
import torch.optim as optim
from dataset import HeteroGraphDataset  # 导入数据集类
from model import Model
import os
import time
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from captum.attr import IntegratedGradients

def write_log(message, log_dir):
    with open(os.path.join(log_dir, 'log_train.txt'), 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    print(message)

def save_importance_to_csv(importance_results, epoch, log_dir):
    records = []
    for item in importance_results:
        sample_name = item['sample_name']
        importance_input = item['importance_input']

        input_flat = importance_input.flatten()

        record = {
            'sample_name': sample_name,
            'epoch': epoch + 1
        }
        for idx, val in enumerate(input_flat):
            record[f'input_feature_{idx}'] = val

        records.append(record)

    df = pd.DataFrame(records)
    csv_filename = os.path.join(log_dir, f'importance_epoch_{epoch + 1}.csv')
    df.to_csv(csv_filename, index=False)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("creating dataset")
    win_size = 5
    batch_size = 128
    n_layer = 2
    num_epochs = 50

    # 创建保存文件的目录
    run_dir = f'run_win{win_size}_batch{batch_size}_nlayer{n_layer}_epoch{num_epochs}'
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    dataset = HeteroGraphDataset("data/20-21data_encoded.csv", win_size)
    print("creating dataset done..... loading data....")

    model_save_path = ""
    if not os.path.exists('run'):
        os.makedirs('run')
    model = Model(residual=False, n_layer=n_layer).to(device)  # Adjusted neighbour_size

    optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.00000001)

    def integrated_gradients_analysis(model, inputs_input, edge_indices, label):
        model.eval()
        inputs_input = inputs_input.to(device).requires_grad_()

        def model_forward(inputs_input):
            output = model(inputs_input, edge_indices)
            return output

        ig = IntegratedGradients(model_forward)
        attributions = ig.attribute(inputs=(inputs_input), target=None, n_steps=50)

        attributions_input = attributions

        importance_input = attributions_input.detach().cpu().numpy()

        return importance_input

    def train(data_list, edge_index_list, sample_names_batch, epoch, log_dir):
        model.train()
        total_loss = 0
        importance_results = []

        for i, (data, edge_indices, sample_name) in enumerate(zip(data_list, edge_index_list, sample_names_batch)):
            # 确保 data 是一个 Data 对象，且其属性是 Tensor
            data = data.to(device)  # 直接使用Data对象
            edge_indices = [edge_index.to(device) for edge_index in edge_indices]
            
            inputs_input = data.input.clone().detach().to(device)
            label = data.star.float().to(device)

            inputs_input.requires_grad_()

            optimizer.zero_grad()
            output = model(inputs_input, edge_indices)
            loss = F.mse_loss(output, label.view_as(output))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            importance_input = integrated_gradients_analysis(
                model, inputs_input, edge_indices, label
            )

            importance_results.append({
                'sample_name': sample_name,
                'importance_input': importance_input,
            
            })


        save_importance_to_csv(importance_results, epoch, log_dir)

        return total_loss / len(data_list)

    def test(data_list, edge_index_list, sample_names):
        model.eval()
        total_loss = 0
        outputs = []
        labels = []
        sample_names_list = []

        with torch.no_grad():
            for data, edge_indices, sample_name in zip(data_list, edge_index_list, sample_names):
                # 确保 data 是一个 Data 对象，且其属性是 Tensor
                data = data.to(device)  # 直接使用Data对象
                edge_indices = [edge_index.to(device) for edge_index in edge_indices]
                
                inputs_input = data.input
                output = model(inputs_input, edge_indices)
                label = data.star.float().view_as(output)
                loss = F.mse_loss(output, label)
                total_loss += loss.item()
                outputs.append(output.item())
                labels.append(label.item())
                sample_names_list.append(sample_name)

        return total_loss / len(data_list), outputs, labels, sample_names_list

    best_loss = float('inf')

    # 获取数据并保证返回的是Data对象列表
    train_data_list, train_edge_index_list = dataset.process_data(pd.read_csv("data/20-21data_encoded.csv"))
    test_data_list, test_edge_index_list = dataset.process_data(pd.read_csv("data/20-21data_encoded.csv"))

    train_sample_names = [f"{i}" for i in range(len(train_data_list))]
    test_sample_names = [f"{i}" for i in range(len(test_data_list))]

    best_loss = float('inf')

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        epoch_start_time = time.time()
        epoch_loss = 0
        num_batches = 0

        # 使用批次进行训练
        for current_idx in range(0, len(train_data_list), batch_size):
            batch_sample_names = train_sample_names[current_idx:current_idx + batch_size]
            data_list = []
            edge_index_list = []

            for sample_name in batch_sample_names:
                sample_idx = int(sample_name)
                data_list.append(train_data_list[sample_idx])  # 这里直接是 Data 对象
                edge_index_list.append(train_edge_index_list[sample_idx])

            loss = train(data_list, edge_index_list, batch_sample_names, epoch, run_dir)
            epoch_loss += loss
            num_batches += 1

            if loss < best_loss:
                best_loss = loss
                if model_save_path:
                    try:
                        os.remove(model_save_path)
                    except OSError:
                        pass
                model_save_path = f"run/epoch_{epoch + 1}_Loss_{best_loss:.4f}.pt"
                torch.save(model.state_dict(), model_save_path)

        avg_epoch_loss = epoch_loss / num_batches
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        write_log(f"Epoch: {epoch + 1}, Average Loss: {avg_epoch_loss:.4f}, Duration: {epoch_duration:.2f} seconds", run_dir)

    # 测试阶段
    all_outputs = []
    all_labels = []
    all_sample_names = []

    for current_idx in range(0, len(test_data_list), batch_size):
        batch_sample_names = test_sample_names[current_idx:current_idx + batch_size]
        data_list = []
        edge_index_list = []

        for sample_name in batch_sample_names:
            sample_idx = int(sample_name)
            data_list.append(test_data_list[sample_idx])  # 这里直接是 Data 对象
            edge_index_list.append(test_edge_index_list[sample_idx])

        test_loss, outputs, labels, sample_names = test(data_list, edge_index_list, batch_sample_names)
        all_outputs.extend(outputs)
        all_labels.extend(labels)
        all_sample_names.extend(sample_names)

    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_labels = np.where(all_labels == 0, 1e-7, all_labels)

    mse = mean_squared_error(all_labels, all_outputs)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_outputs)
    mape = np.mean(np.abs((all_labels - all_outputs) / all_labels)) * 100
    r2 = r2_score(all_labels, all_outputs)

    write_log(f"Test MSE: {mse:.4f}", run_dir)
    write_log(f"Test RMSE: {rmse:.4f}", run_dir)
    write_log(f"Test MAE: {mae:.4f}", run_dir)
    write_log(f"Test MAPE: {mape:.4f}%", run_dir)
    write_log(f"Test R²: {r2:.4f}", run_dir)

    # 绘制测试结果
    sample_names_to_plot = all_sample_names[:30]
    outputs_to_plot = all_outputs[:30]
    labels_to_plot = all_labels[:30]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_names_to_plot, outputs_to_plot, label='pre star', color='blue', marker='o')
    plt.plot(sample_names_to_plot, labels_to_plot, label='real star', color='red', marker='x')
    plt.xlabel('Sample Name')
    plt.ylabel('Value')
    plt.title('Test Sample Output and Label')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'test_output_label_plot.png'))
    plt.show()

if __name__ == '__main__':
    main()
