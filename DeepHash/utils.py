import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def datasetconfig(config):
    if config["dataset"] == "cifar":
        config["topK"] = -1
        config["class_num"] = 10
    elif config["dataset"] == "nuswide":
        config["topK"] = 5000
        config["class_num"] = 21
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["class_num"] = 80

    if config["dataset"] == "cifar":
        config["data_path"] = "./dataset/cifar/"
    if config["dataset"] == "nuswide":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "coco":
        config["data_path"] = "./dataset/COCO_2014/"

    config["data"] = {
        "train_set": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

# 改变图片形状
def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


# 处理cifar数据集
def cifar_dataset(config):
    batch_size = config["batch_size"]
    train_size = 500
    test_size = 100
    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = 'dataset/cifar/'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))
    # cifar数据集有10类，每类抽500做训练，每类抽100做测试，其余做数据库
    first = True
    for label in range(10):
        index = np.where(L == label)[0]
        N = index.shape[0]
        # 排序
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False



    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=1)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=1)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  num_workers=1)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(config):
    if config["dataset"] == "cifar":
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=1, drop_last=True)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


# 获取测试所需的数据VCGDH
def compute_result(dataloader, net):
    real_hash_code, labels = [], []
    for img, label, _ in tqdm(dataloader):
        img = img.cuda()
        label = label.float().cuda()
        labels.append(label.cpu().detach())
        hash_code, cls = net(img)
        hash_code = hash_code.sign()
        real_hash_code.append(hash_code.cpu().detach())

    return torch.cat(real_hash_code), torch.cat(labels)

# 获取测试所需的数据（其他模型）
def compute_result_old(dataloader, net, device):
    real_hash_code, labels = [], []
    for img, label, _ in tqdm(dataloader):
        img = img.to(device)
        label = label.float().to(device)
        labels.append(label.cpu().detach())
        hash_code = net(img)
        hash_code = hash_code.sign()
        real_hash_code.append(hash_code.cpu().detach())

    return torch.cat(real_hash_code), torch.cat(labels)

# 计算汉明距离
def CalcHammingDist(b1, b2):

    q = b2.shape[1]
    distH = 0.5 * (q - np.dot(b1, b2.transpose()))
    return distH

# 计算mAP
def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_tmp = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_tmp
    topkmap = topkmap / num_query
    return topkmap


# 画 P-R 曲线
def pr_curve(rB, qB, retrievalL, queryL, topK):

    n_query = qB.shape[0]
    if topK == -1 or topK > rB.shape[0]:
        topK = rB.shape[0]

    Gnd = (np.dot(queryL, retrievalL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qB, rB))

    P, R = [], []
    for k in range(1, topK + 1):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]

            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)

            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R

