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
        "m": 1,
        "p": 0.5,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        "info": "[DPN]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 100,
        "net": AlexnetFc,
        "dataset": "cifar",
        "epoch": 150,
        "save_path": "save/DPN",
        "PR_save": "PR_save",
        "bit": 48,

    }
    config = datasetconfig(config)
    return config


# -----------  损失函数 ------------
class DPNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPNLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "coco"}
        self.target_vectors = self.get_target_vectors(config["n_class"], bit, config["p"]).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.m = config["m"]
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        if "-T" in config["info"]:
            # Ternary Assignment
            u = (u.abs() > self.m).float() * u.sign()

        t = self.label2center(y)
        return (self.m - u * t).clamp(0).mean()

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.target_vectors[y.argmax(axis=1)]
        else:
            # for multi label, use the same strategy as CSQ
            center_sum = y @ self.target_vectors
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # Random Assignments of Target Vectors
    def get_target_vectors(self, n_class, bit, p=0.5):
        target_vectors = torch.zeros(n_class, bit)
        for k in range(20):
            for index in range(n_class):
                ones = torch.ones(bit)
                sa = random.sample(list(range(bit)), int(bit * p))
                ones[sa] = -1
                target_vectors[index] = ones
        return target_vectors

    # Adaptive Updating
    def update_target_vectors(self):
        self.U = (self.U.abs() > self.m).float() * self.U.sign()
        self.target_vectors = (self.Y.t() @ self.U).sign()

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
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = cifar_dataset(config)
    bit = config["bit"]
    config["num_train"] = num_train
    config["device"] = device
    criterion = DPNLoss(config, bit)
    # 训练阶段
    model = config["net"](bit).cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    train(train_loader)
    # 测试阶段
    map_test(test_loader, dataset_loader, device)





