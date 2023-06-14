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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
gpus = [0, 1]
gpu_usg = True
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")



# -----------  模型参数设置 ------------
def get_config():
    config = {
        "alpha": 0.1,
        "use_square_clamp": True,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}, "lr_type": "step"},
        # "optimizer": {"type": optim.Adam, "lr": 1e-5},
        "info": "[VCGDH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 100,
        "net": VCGDH,
        "dataset": "cifar",
        "epoch": 100,
        "save_path": "save/VCGDH",
        "PR_path":"PR_save",
        "bit": 32,

    }
    config = datasetconfig(config)
    return config



# ----------- 训练 ------------

def train(data_loader, test_loader, dataset_loader):
    best = 0
    for epoch in range(config["epoch"]):
        print("开始第%i个epoch的训练" % epoch)
        torch.cuda.empty_cache()
        model.eval()

        model.train()
        train_loss = 0
        for image, label, ind in data_loader:
            image = image.cuda()
            label = label.float().cuda()
            optimizer.module.zero_grad()
            real_hash_code, cla = model(image)
            #定义相似矩阵S
            S = (label @ label.T).cpu()
            S = torch.from_numpy(np.where(S > 0, 1.0, 0.0)).cuda()
            S_num = S.numel()
            S_onenum = S.norm(1)
            S_zeronum = S_num - S_onenum

            # binary_hash_code = ((real_hash_code.sign()).add(1)).div(2)

            # 成对相似性的权重
            w_pair = torch.where(S > 0.5, S_num/S_onenum, S_num/S_zeronum)
            # # print("成对相似性权重")
            # # print(w_pair)
            #
            # 计算余弦相似性
            cos_sim = torch.cosine_similarity(real_hash_code.unsqueeze(1), real_hash_code.unsqueeze(0), dim=-1)
            # print(cos_sim)
            q = (cos_sim.add(1)).div(2)
            # # print("q:")
            # # print(q)
            #
            # 相似性损失函数
            Loss_sim_first = w_pair*q*torch.log(2*(q.div(q.add(S))))
            Loss_sim_second = w_pair*S*torch.log((2*(S.div(q.add(S)))).add(1e-6))
            Loss_sim = torch.mean(Loss_sim_first + Loss_sim_second)
            # # print("相似性损失函数:")
            # # print(Loss_sim)


            # 语义损失
            c_t = torch.sum(S, dim=1)
            label_max_value, label_max_indx = torch.max(label, dim=1)
            cla_max_value, cla_max_indx = torch.max(cla, dim=1)
            tmp = torch.zeros(class_num)
            c_tf = torch.zeros(label.size(0)).to(device)
            for i in range(label.size(0)):
                if label_max_indx[i]==cla_max_indx[i]:
                    tmp[label_max_indx[i]] = tmp[label_max_indx[i]]+1
            for i in range(label.size(0)):
                c_tf[i] = c_t[i] - tmp[label_max_indx[i]]
            w_sem = c_tf.div(c_t)
            # # print("语义权重：")
            # # print(w_sem)
            # # print(tmp)
            # # print(c_t)
            # # print(c_tf)
            # # print(label_max_indx)
            # # print(cla_max_indx)
            #Loss_sem = torch.mean(w_sem*torch.sum(label*torch.log(cla)*(-1), dim=1))
            Loss_sem = torch.mean(torch.sum(label * torch.log(cla) * (-1), dim=1))
            # print("语义损失函数：")
            # print(Loss_sem)
            Loss = Loss_sim + Loss_sem
            Loss.backward()

            train_loss += Loss
            # loss = criterion(real_hash_code, label.float(), ind, config)
            # train_loss += loss.item()
            # loss.mean().backward()
            optimizer.module.step()

        train_loss = train_loss / len(data_loader)
        print("Loss:%.3f" % train_loss)

    torch.save(model.state_dict(), os.path.join(config["save_path"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-model.pt"))



def map_test(test_loader, dataset_loader):
    with torch.no_grad():
        model_dict = torch.load(os.path.join(config["save_path"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-model.pt"))
        net = config["net"](bit, class_num).to(device)
        net = nn.DataParallel(net, device_ids=gpus)
        net.load_state_dict(model_dict, strict=False)
        net = net.module
        net.eval()

        test_binary, test_label = compute_result(test_loader, net)
        dataset_binary, dataset_label = compute_result(dataset_loader, net)

        test_binary = test_binary.numpy()
        test_label = test_label.numpy()
        dataset_binary = dataset_binary.numpy()
        dataset_label = dataset_label.numpy()
        mAP = CalcTopMap(dataset_binary, test_binary, dataset_label, test_label, config["topK"])
        print(f"mAP:{mAP} " + config["info"] + str(config["bit"]))
        P, R = pr_curve(dataset_binary, test_binary, dataset_label, test_label, config["topK"])
        np.save(os.path.join(config["PR_path"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-P.npy"), P)
        np.save(os.path.join(config["PR_path"], config["dataset"] + "-" + str(config["batch_size"]) + "-" + str(config["bit"]) + "-" + str(config["epoch"]) + "-" + config["info"] + "-R.npy"), R)



if __name__ == '__main__':
    config = get_config()
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    bit = config["bit"]
    config["num_train"] = num_train
    class_num = config["class_num"]
    #criterion = VCGDHLoss(config, bit)

    # 训练阶段
    model = nn.DataParallel(config["net"](bit, class_num), device_ids=gpus).cuda()
    optimizer = config["optimizer"]["type"](model.parameters(), lr=config["optimizer"]["optim_params"]["lr"])
    optimizer = nn.DataParallel(optimizer,device_ids=gpus)
    train(train_loader, test_loader, dataset_loader)
    # 测试阶段
    map_test(test_loader, dataset_loader)