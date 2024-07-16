import os
import pdb
import torch
import torchvision
import pandas as pd
import plotext as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader

def try_device():
    return torch.device('mps')

def accuracy(outputs, labels):
    return (outputs.argmax(dim=-1) == labels).float().mean()


# 模型
class MyModel(torch.nn.Module):
    def __init__(self, out_label):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, out_label)

    def forward(self, X):
        return self.resnet(X)


# 预处理
class MyDataset(Dataset):
    def __init__(self, labels, img_dir, mode=None, target_transform=None):
        super().__init__()
        self.img_labels = labels
        self.img_dir = img_dir
        self.target_transform = target_transform
        if mode == 'train':
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.RandomHorizontalFlip(p=0.25),
                torchvision.transforms.RandomVerticalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'val':
            preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.transform = preprocess

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir[idx])
        with Image.open(img_path) as im:
            image = im
            label = self.img_labels.iloc[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

def train(epochs, train_dataloader, model, loss, optimizer):
    for epoch in range(epochs):

        losses = []
        val_accs = []
        train_accs = []

        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(try_device()), labels.to(try_device())

            optimizer.zero_grad()

            outputs = model(inputs)
            l = loss(outputs, labels)
            l.backward()

            optimizer.step()

            train_accs.append(accuracy(outputs, labels))
            losses.append(l)
            del inputs, labels

        # log train loss
        loss_mean = sum(losses) / len(losses)
        train_accuracy = sum(train_accs) / len(train_accs)
        print('epoch:', epoch, 'loss:', loss_mean)
        print('epoch:', epoch, 'train accuracy:', train_accuracy)

        for i, data in enumerate(val_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(try_device()), labels.to(try_device())

            outputs = net(inputs)
            val_accs.append(accuracy(outputs, labels))
            del inputs, labels

        # log val
        val_accuracy = sum(val_accs) / len(val_accs)
        print('epoch:', epoch, 'val accuracy:', val_accuracy)

        torch.save(net.state_dict(), 'checkpoint_' + str(epoch))

#        plt.plot(steps, losses)
#        plt.plot(steps, train_accuracys)
#        plt.grid(True)
#        plt.show()
#        plt.clf()


if __name__ == '__main__':
    ### 导入数据
    # 测试数据
    test_data = pd.read_csv("./test.csv")
    # 训练数据
    all_data = pd.read_csv("./train.csv")
    train_data = all_data.sample(n=int(len(all_data) * 0.9), ignore_index=True)
    val_data = all_data.sample(n=int(len(all_data) * 0.1), ignore_index=True)
    # 获取类别数量
    classes = all_data['label'].unique().tolist()
    print("train_data shape:", all_data.shape, "test_data shape:", test_data.shape, "label size:", len(classes))


    bs = 64
    lr = 0.1
    num_epochs = 50

    # 损失函数
    criterion = torch.nn.CrossEntropyLoss()
    
    net = MyModel(len(classes)).to(device=try_device())
    # 优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    target_transform = lambda y: torch.tensor(classes.index(y))
    
    training_data = MyDataset(train_data['label'], train_data['image'], 'train', target_transform)
    train_dataloader = DataLoader(training_data, batch_size=bs, shuffle=True)
    
    val_data = MyDataset(val_data['label'], val_data['image'], 'val', target_transform)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False)
    
    test_data = MyDataset(test_data['image'], test_data['image'], 'test')
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)
    print("train_data length:", len(training_data), "test_data length:", len(test_data), "val_data length:", len(val_data))

    train(num_epochs, train_dataloader, net, criterion, optimizer)

