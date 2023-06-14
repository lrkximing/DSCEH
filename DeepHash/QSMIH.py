from utils import *
from model.model import *
import torch
import torch.optim as optim
import random
import os


# ----------- 随机种子 ----------
seed = 1
# 设置种子，使神经网络参数初始化
random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
dtype = torch.cuda.FloatTensor

# -----------  GPU设置 ------------
gpu_usg = True
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")




# -----------  模型参数设置 ------------
def get_config():
    config = {
        "alpha": 0.01,
        "sigma": 0,
        "use_square_clamp": True,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[QSMIH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexnetFc,
        "dataset": "cifar",
        "epoch": 100,
        "save_path": "save/QSMIH",
        "PR_save": "PR_save",
        "bit": 48,

    }
    config = datasetconfig(config)
    return config


# -----------  损失函数 ------------
class QSMIHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(QSMIHLoss, self).__init__()

    def forward(self, u, y, ind, config):


        u = u / (torch.sqrt(torch.sum(u ** 2, dim=1, keepdim=True)) + 1e-8)
        Y = torch.mm(u, u.t())
        Y = 0.5 * (Y + 1)

        # Get the indicator matrix \Delta
        # D = (y.view(y.shape[0], 1) == y.view(1, y.shape[0]))
        D = (y @ y.t() > 0).float()

        M = D.size(1) ** 2 / torch.sum(D)

        if config["use_square_clamp"]:
            Q_in = (D * Y - 1) ** 2
            Q_btw = (1.0 / M) * Y ** 2
            # Minimize clamped loss
            L_QSMI = Q_in + Q_btw
        else:
            Q_in = D * Y
            Q_btw = (1.0 / M) * Y
            # Maximize QMI/QSMI
            L_QSMI = Q_btw - Q_in

        L_QSMI = L_QSMI.mean()
        L_hash = config["alpha"] * (u.abs() - 1).abs().mean()
        return L_QSMI + L_hash
# ----------- 训练 ------------

def train(data_loader):

    for epoch in range(config["epoch"]):
        print("开始第%i个epoch的训练" % epoch)
        torch.cuda.empty_cache()
        model.eval()
        model.train()
        train_loss = 0

        for image, label, ind in data_loader:

            image = image.to(device)
            label = label.float().to(device)
            optimizer.zero_grad()
            real_hash_code = model(image)
            loss = criterion(real_hash_code, label.float(), ind, config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(data_loader)
        print("Loss:%.3f" % train_loss)
    torch.save(model.state_dict(), os.path.join(config["save_path"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-model.pt"))



def map_test(test_loader, dataset_loader, device):
    net = config["net"](bit).cuda()
    net.load_state_dict(torch.load(os.path.join(config["save_path"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-model.pt")))
    net.eval()

    test_binary, test_label = compute_result_old(test_loader, net, device)
    dataset_binary, dataset_label = compute_result_old(dataset_loader, net, device)

    test_binary = test_binary.numpy()
    test_label = test_label.numpy()
    dataset_binary = dataset_binary.numpy()
    dataset_label = dataset_label.numpy()
    mAP = CalcTopMap(dataset_binary, test_binary, dataset_label, test_label, config["topK"])
    print(f"mAP:{mAP}" + " " + config["info"])
    # P, R = pr_curve(dataset_binary, test_binary, dataset_label, test_label, config["topK"])
    # np.save(os.path.join(config["PR_save"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-P.npy"), P)
    # np.save(os.path.join(config["PR_save"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-R.npy"), R)



if __name__ == '__main__':
    config = get_config()
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    bit = config["bit"]
    config["num_train"] = num_train
    config["device"] = device
    criterion = QSMIHLoss(config, bit)
    # 训练阶段
    model = config["net"](bit).cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    train(train_loader)
    # 测试阶段
    map_test(test_loader, dataset_loader, device)





