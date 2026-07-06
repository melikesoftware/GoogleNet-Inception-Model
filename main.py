
import os

from pathlib import Path

import matplotlib.pyplot as plt

from torchvision import datasets
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader,Subset

import torch
from torchinfo import summary
from PIL import Image

flower_path=Path("data/")
image_path=flower_path/"Animals-10"


device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

torch.manual_seed(42)



def check_data(dir_path):
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"# of direcitories: {len(dirnames)} and {len(filenames)} images in {dirpath}")

check_data(image_path)



train_transform=transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize( mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

test_transform=transforms.Compose([


    transforms.Resize((224, 224)),


    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_ratio=0.2

train_data=datasets.ImageFolder(root=image_path,transform=train_transform)
test_data=datasets.ImageFolder(root=image_path,transform=test_transform)

total_data=len(train_data)
total_test=int(total_data*test_ratio)
total_train=total_data-total_test

print(total_data)
print(total_train)
print(total_test)

g=torch.Generator().manual_seed(42)
perm=torch.randperm(total_data,generator=g).tolist()

train_idx=perm[:total_train]
test_idx=perm[total_train:]
print(len(train_idx))
print(len(test_idx))

train_dataset=Subset(train_data,train_idx)
test_dataset=Subset(test_data,test_idx)

print(len(train_dataset))
print(len(test_dataset))

BATCH_SIZE = 32



train_dataLoader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_dataLoader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

print(len(train_dataLoader))
print(len(test_dataLoader))

class_names=train_data.classes
print(class_names)
print(len(class_names))


#


#Çoğu CNN'de kernel tek sayıysa padding genellikle =(kernel−1)/2


class Inception(nn.Module):
    def __init__(self,in_channels,c1,c3_reduce,c3,c5_reduce,c5,pool_proj):
        super().__init__()
        self.branch1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=c1,kernel_size=1,padding=0,stride=1),
            nn.ReLU()
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=c3_reduce,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3_reduce, out_channels=c3, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.branch3=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c5_reduce, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c5_reduce,out_channels=c5,kernel_size=5,padding=2,stride=1),
            nn.ReLU()
        )
        self.branch4=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=in_channels,out_channels=pool_proj,kernel_size=1),
            nn.ReLU()
        )

    def forward(self,x):
        b1=self.branch1(x)
        b2=self.branch2(x)
        b3=self.branch3(x)
        b4=self.branch4(x)

        return torch.cat([b1,b2,b3,b4],dim=1)



class Datas(nn.Module):
    def __init__(self,input_shape:int,hidden_units:int,num_classes:int):
        super().__init__()

        self.block1=nn.Sequential(
            nn.Conv2d(in_channels=input_shape,out_channels=hidden_units,kernel_size=7,padding=3,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units*3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = Inception(in_channels=hidden_units*3,c1=64,c3_reduce=96,c3=128,c5_reduce=16,c5=32,pool_proj=32)

        self.inception3b = Inception(in_channels=hidden_units * 4,c1=128,c3_reduce=128,c3=192,c5_reduce=32,c5=96,pool_proj=64)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception4a = Inception(in_channels=480,c1=192,c3_reduce=96,c3=208,c5_reduce=16,c5=48,pool_proj=64)

        self.inception4b = Inception(in_channels=hidden_units * 8,c1=160,c3_reduce=112,c3=224,c5_reduce=24,c5=64,pool_proj=64)

        self.inception4c = Inception(in_channels=hidden_units * 8,c1=128,c3_reduce=128,c3=256,c5_reduce=24,c5=64,pool_proj=64)

        self.inception4d = Inception(in_channels=hidden_units * 8,c1=112,c3_reduce=144,c3=288,c5_reduce=32,c5=64,pool_proj=64)
        self.inception4e =Inception(in_channels=528,c1=256,c3_reduce=160,c3=320,c5_reduce=32,c5=128,pool_proj=128)

        self.block3=nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )



        self.inception5a = Inception(in_channels=hidden_units * 13,c1=256,c3_reduce=160,c3=320,c5_reduce=32,c5=128,pool_proj=128)

        self.inception5b = Inception(in_channels=hidden_units * 13,c1=384,c3_reduce=192,c3=384,c5_reduce=48,c5=128,pool_proj=128)


        self.block4=nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=1024,out_features=num_classes)
        )

    def forward(self, x):
            x = self.block1(x)
            x = self.inception3a(x)
            x = self.inception3b(x)
            x = self.block2(x)
            x = self.inception4a(x)
            x = self.inception4b(x)
            x = self.inception4c(x)
            x = self.inception4d(x)
            x = self.inception4e(x)
            x = self.block3(x)

            x = self.inception5a(x)
            x = self.inception5b(x)
            x=self.block4(x)


            return x



torch.manual_seed(42)
datas_model=Datas(input_shape=3,hidden_units=64,num_classes=len(class_names)).to(device)



summary(datas_model,input_size=[32,3,224,224],device=device)

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=datas_model.parameters(),lr=0.0001)

torch.manual_seed(42)

epochs=20

train_loss_values=[]
test_loss_values=[]
train_acc_values=[]
test_acc_values=[]

for epoch in range(epochs):
    total_loss=0
    total_acc=0

    for batch,(X,y) in enumerate(train_dataLoader):
        X = X.to(device)
        y = y.to(device)
        datas_model.train()
        y_pred=datas_model(X)
        loss=loss_function(y_pred,y)
        total_loss+=loss.item()
        y_pred_class=torch.argmax(y_pred,dim=1)
        total_acc+=(y_pred_class==y).sum().item()*100/len(y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(batch%200==0):
            print(f"Looked at:{batch}")

    total_loss/=len(train_dataLoader)
    total_acc/=len(train_dataLoader)

    train_loss_values.append(total_loss)
    train_acc_values.append(total_acc)

    datas_model.eval()
    test_loss=0
    test_acc=0

    with torch.inference_mode():
        for (X,y) in test_dataLoader:
            X = X.to(device)
            y = y.to(device)
            logits=datas_model(X)
            loss1=loss_function(logits,y)
            test_loss+=loss1.item()

            logits_class=torch.argmax(logits,dim=1)
            test_acc+=(logits_class==y).sum().item()*100/len(logits)

        test_loss/=len(test_dataLoader)
        test_acc/=len(test_dataLoader)

        test_loss_values.append(test_loss)
        test_acc_values.append(test_acc)

        print(
        f"Train loss:{total_loss} ,Train accuracy:{total_acc},Test loss:{test_loss}, Test accuracy:{test_acc}")



epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_values, label='Train Loss')
plt.plot(epochs_range, test_loss_values, label='Test Loss')
plt.title('Train and Test Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc_values, label='Train Accuracy')
plt.plot(epochs_range, test_acc_values, label='Test Accuracy')
plt.title('Train and Test Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

image_transform=transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])


prediction_example=Path("datas/")
prediction_ids= ("butterfly.png","chicken.png","cow.png")

datas_model.eval()
with torch.inference_mode():
    for datas in prediction_ids:
        imag_path=prediction_example/datas
        image=Image.open(img_path).convert("RGB")
        image = image_transform(image).unsqueeze(0).to(device)
        pred = datas_model(image)
        pred_idx = torch.argmax(pred, dim=1)
        print(f"{image_path} -> {class_names[pred_idx]}")



































