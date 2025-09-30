from matplotlib import pyplot as plt
from matplotlib import colors
import os
import torch
def save_visualization(pred, save_path, time,var):
    """保存可视化图片"""
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))

    # 预测结果（t2m）
    a0 = axs.imshow(pred.detach().flip(0).cpu().numpy())
    axs.set_title(f'{var} - {time}')
    axs.axis('off')
    fig.colorbar(a0, ax=axs, orientation='horizontal', shrink=0.8, aspect=16, extend='both')

    

    plt.tight_layout()
    path=os.path.join(save_path, f'{var}_{time}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 示例数据
    import json
    with open('sshf_mean_std.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    data_folder = "/sharefiles2/guoyixin/datasets/new_weather_tensors/1979"
    i= "1979-3970.pt"  # 假设这是一个预测结果文件名
    pred = torch.load(os.path.join(data_folder, i))  # 假设预测结果是一个 128x256 的张量
    save_path = "./visualizations"
    os.makedirs(save_path, exist_ok=True)
    time = "19793970"
    var = "sshf_1"
    
    # save_visualization((pred[69]-torch.tensor(data['mean_temp']))/torch.tensor(data['std_temp']), save_path, time, var)
    save_visualization((pred[69]-(-41173.4609375))/170182.640625, save_path, time, var)
    print(f"Visualization saved to {save_path}/{var}_{time}.png")