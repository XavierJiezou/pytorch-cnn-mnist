#%% 数据说明
'''
测试数据是用Windows上的画图软件手写的10个数字
'''
#%% 导入模块
import os
import torch
from PIL import Image
from model import CNN
import matplotlib.pyplot as plt
from torchvision import transforms
#%% 数据准备
path = './test/'
imgs = []
labels = []
for name in sorted(os.listdir(path)):
    img = Image.open(path+name).convert('L')
    img = transforms.ToTensor()(img)
    imgs.append(img)
    labels.append(int(name[0]))
imgs = torch.stack(imgs, 0)
#%% 加载模型
model = CNN()
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()
#%% 测试模型
with torch.no_grad():
    output = model(imgs)
#%% 打印结果
pred = output.argmax(1)
true = torch.LongTensor(labels)
print(pred)
print(true)
#%% 结果显示
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title(f'pred {pred[i]} | true {true[i]}')
    plt.axis('off')
    plt.imshow(imgs[i].squeeze(0), cmap='gray')
plt.savefig('test.png')
plt.show()