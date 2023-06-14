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
        "alpha": 0.1,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[Hashnet]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexnetFc,
        "dataset": "cifar",
        "epoch":  50,
        "save_path": "save/Hashnet",
        "PR_save": "PR_save",
        "bit": 48,

    }
    config = datasetconfig(config)
    return config


# -----------  损失函数 ------------
class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

        self.scale = 1

    def forward(self, u, y, ind, config):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


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
    criterion = HashNetLoss(config, bit)
    # 训练阶段
    model = config["net"](bit).cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    train(train_loader)
    # 测试阶段
    map_test(test_loader, dataset_loader, device)





