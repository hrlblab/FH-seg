import matplotlib.pyplot as plt
import pandas as pd

def save_plot(data, title, xlabel, ylabel, file_path, legend_label):
    """生成并保存图表。
    :param data: 数据列表。
    :param title: 图表标题。
    :param xlabel: X轴标签。
    :param ylabel: Y轴标签。
    :param file_path: 图片保存路径。
    :param legend_label: 图例标签。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=legend_label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()  # 关闭图表以避免内存泄露

def save_data_to_excel(epochs, data, file_path):
    """将数据保存为Excel文件。
    :param epochs: Epoch数列表。
    :param data: 相应数据列表。
    :param file_path: 文件保存路径。
    """
    df = pd.DataFrame({
        'Epoch': epochs,
        'Value': data
    })
    df.to_excel(file_path, index=False)


import os


def save_learning_rate_or_loss_data(epoch, lr_changes, base_dir, sub_dir, file_name_prefix, chart_title,y_label='Learning Rate',legend_label ='Learning Rate' ):
    """
    保存和可视化学习率数据。

    :param epoch: 当前 epoch 索引。
    :param lr_changes: 学习率变化列表。
    :param base_dir: 数据和图形保存的基础目录。
    :param sub_dir: 子目录，用于进一步细化保存路径。
    :param file_name_prefix: 保存文件的前缀，用于生成文件名。
    :param chart_title: 图表的标题。
    """
    # 构建完整的文件保存路径
    full_path = os.path.join(base_dir, sub_dir)
    if not os.path.exists(full_path):
        os.makedirs(full_path)  # 如果路径不存在，创建它

    plot_path = os.path.join(full_path, f"{file_name_prefix}_Epoch_{epoch}.png")
    excel_path = os.path.join(full_path, f"{file_name_prefix}_Changes_Epoch_{epoch}.xlsx")
    # 生成并保存图形
    save_plot(lr_changes, chart_title, 'Epoch', y_label, plot_path, legend_label)

    # 保存学习率数据到 Excel
    epochs_list = list(range(epoch + 1))  # 创建一个包含当前 epoch 的列表
    save_data_to_excel(epochs_list, lr_changes, excel_path)

    # 打印保存信息
    print(f"Visualizations and data saved for epoch {epoch} at {full_path}")
