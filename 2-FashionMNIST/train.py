# This Python file uses the following encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import tqdm
from matplotlib import pyplot as plt
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

transform_train=transforms.Compose(
     [transforms.RandomHorizontalFlip(),
     transforms.ToTensor()]
)
#只对训练的图片进行处理，主要进行了左右翻转

mnist_train_set=torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=transform_train)
mnist_test_set=torchvision.datasets.FashionMNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())
#单通道（灰度图像），28*28pixel
train_loader = torch.utils.data.DataLoader(mnist_train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(mnist_test_set, batch_size=128, shuffle=False, num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,64,1,padding=1)
        #size 128(batch_size)*64(out_channel)*30*30[28+2*padding-kernel_x_size+1]
        self.conv2=nn.Conv2d(64,64,3,padding=1)
        self.pool1=nn.MaxPool2d(kernel_size=(2, 2))
        #采用maxpool2d对数据进行池化，采集特征
        self.bn1=nn.BatchNorm2d(64)
        #进行正则化
        self.relu1=nn.Sigmoid()
        #激活函数，使得分布映射到0-1区间
        self.fc1=nn.Linear(128*15*15,10)
        #10分类问题，进行简单的线性拼接

        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3,padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        #padding在外围一圈填充
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.fc5 = nn.Linear(128*8*8,512)
        self.drop1 = nn.Dropout2d()
        #将输入的每个数值按（伯努利分布）概率赋值为0
        self.fc6 = nn.Linear(512,10)



    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        #x = self.fc1(x.view(-1,128*15*15))

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

    
        x = x.view(-1,128*8*8)
        #使得x的维度进行变换，便于后续处理
        x = F.relu(self.fc5(x))
        x = self.drop1(x)
        x = self.fc6(x)

        # return F.softmax(x)
        return x

    def train(self,all_epoch,device,epochs=500):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        #optimizer = optim.SGD(self.parameters(),lr=0.001)

        path = 'weights.tar'
        # initepoch = 0
        loss = nn.CrossEntropyLoss()
        epoch_num=[]
        per_epoch_loss=[]
        acc_array=[]
        OverallEpoch_loss = 0

        for epoch in range(epochs):  # loop over the dataset multiple times
            with tqdm.tqdm(total=len(train_loader)) as progress:
                progress.set_description('Epoch %i'% epoch)
                
                timestart = time.time()

                running_loss = 0.0
                total = 0
                correct = 0

                num_iteration = 0
                acc = 0
                
                for i, data in enumerate(train_loader, 0):
                # get the inputs
                    inputs, labels = data
                    inputs, labels = inputs.to(device),labels.to(device)

                # zero the parameter gradients
                    optimizer.zero_grad()

                # forward + backward + optimize
                    outputs = self(inputs)
                    l = loss(outputs, labels)
                    l.backward()
                    optimizer.step()

                # print statistics
                    running_loss += l.item()
           
                    #if(epoch % 10 == 9):
                    _, output_label = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (output_label == labels).sum().item()
                    acc += 100.0 * correct / total
                           
                    progress.set_postfix(loss=l.item())
                    progress.update()

                num_iteration = i
                epoch_num.append(epoch+1)
                per_epoch_loss.append(running_loss/num_iteration)
                OverallEpoch_loss+=running_loss/num_iteration
                #if(epoch % 10 == 9):
                acc /= num_iteration
                acc_array.append(acc) 

            print('epoch %d cost %3f sec' %(epoch,time.time()-timestart))
        

        print("Epoch %d average loss is %3f"%(all_epoch+1,OverallEpoch_loss/epochs))
        print('Finished Training')
        plt.plot(acc_array)
        plt.xlabel("epoch number (*10)")
        plt.ylabel("acc")
        plt.savefig('acc_%d.png'% all_epoch)
        plt.close("all")

        plt.plot(per_epoch_loss)
        plt.xlabel("epoch number")
        plt.ylabel("loss")
        plt.savefig('loss_%d.png'% all_epoch)
        plt.close("all")

    def test(self,device):
        correct = 0
        total = 0
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                test_loss_fn = nn.CrossEntropyLoss()
                test_loss += test_loss_fn(outputs,labels).item()
                test_loss /= len(test_loader)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


        print('Accuracy of the network on the 10000 test images: %.3f %%' % (
                100.0 * correct / total))
        print(f"Avg loss:{test_loss:>8f} \n")

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net()
    net = net.to(device)
    epochs=3
    for epoch in range(epochs):
        net.train(epoch,device,100)
        net.test(device)

    






