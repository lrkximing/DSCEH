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
        "gamma": 20.0,
        "lambda": 0.1,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[DCH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 100,
        "net": AlexnetFc,
        # "dataset": "cifar",
        "dataset": "coco",
        "epoch": 150,
        "save_path": "save/DCH",
        "PR_save": "PR_save",
        "bit": 48,

    }
    config = datasetconfig(config)
    return config


# -----------  损失函数 ------------
class DCHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DCHLoss, self).__init__()
        self.gamma = config["gamma"]
        self.lambda1 = config["lambda"]
        self.K = bit
        self.one = torch.ones((config["batch_size"], bit)).to(config["device"])

    def d(self, hi, hj):
        inner_product = hi @ hj.t()
        norm = hi.pow(2).sum(dim=1, keepdim=True).pow(0.5) @ hj.pow(2).sum(dim=1, keepdim=True).pow(0.5).t()
        cos = inner_product / norm.clamp(min=0.0001)
        # formula 6
        return (1 - cos.clamp(max=0.99)) * self.K / 2

    def forward(self, u, y):
        s = (y @ y.t() > 0).float()

        if (1 - s).sum() != 0 and s.sum() != 0:
            # formula 2
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            # maybe |S1|==0 or |S2|==0
            w = 1

        d_hi_hj = self.d(u, u)
        # formula 8
        cauchy_loss = w * (s * torch.log(d_hi_hj / self.gamma) + torch.log(1 + self.gamma / d_hi_hj))
        # formula 9
        quantization_loss = torch.log(1 + self.d(u.abs(), self.one) / self.gamma)
        return cauchy_loss.mean() + self.lambda1 * quantization_loss.mean()

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
            loss = criterion(real_hash_code, label.float())
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
    print(f"mAP:{mAP} " + config["info"])
    # P, R = pr_curve(dataset_binary, test_binary, dataset_label, test_label, config["topK"])
    # np.save(os.path.join(config["PR_save"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-P.npy"), P)
    # np.save(os.path.join(config["PR_save"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-R.npy"), R)



if __name__ == '__main__':
    config = get_config()
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    bit = config["bit"]
    config["num_train"] = num_train
    config["device"] = device
    criterion = DCHLoss(config, bit)
    # 训练阶段
    model = config["net"](bit).cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    train(train_loader)
    # 测试阶段
    map_test(test_loader, dataset_loader, device)
