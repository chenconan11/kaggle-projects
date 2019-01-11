# Dog Breed Identification
## 1. 项目说明
* 复习Pytorch的使用
* 复习经典的网络结构
* 练习kaggle竞赛
## 2. 文件说明
### 2.1 kaggleImageFolder.py   
定义了 KaggleImageFolder 类，来制作dataset。  
使用说明：
```python
import torchvision.transforms as transforms
from kaggleImageFolder import KaggleImageFolder
root = 'd:/DATA/dog breed'
train_dir = root + '/train'
test_dir = root + '/test'
labels_csv = 'labels.csv'

data_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# train dataset
train_data = KaggleImageFolder(train_dir, labels_csv, transform=data_transform, mode='train', split_p=0.2)
print(len(train_data))
# valid dataset, 采用均匀分布提取
valid_data = KaggleImageFolder(train_dir, labels_csv, transform=data_transform, mode='valid', split_p=0.2)
print(len(valid_data))
# test dataset, 不包含 label 信息
test_data = KaggleImageFolder(test_dir, transform=data_transform, mode='test')
print(len(test_data))
```  
### 2.2 classifier.py  
* AngClassifier类型可以创建Pytorch支持的卷积模型，例如创建vgg16:   
```python
from classifier import AngClassifier
# 创建
vgg16 =  AngClassifier('vgg16', hidden_units=[1024], n_classes=10)
# 保存
vgg16.save_model('vgg16.pth')
# 恢复
model = AngClassifier(load_file='vgg16.pth')
```
* MixClassifier继承了 `nn.Module`, 融合`resnet152`和`densenet169`的特征向量，加`Linear layer`形成模型做fine-tuning。
```python
from classifier import MixClassifier
# 创建
model = MixClassifier(n_classes=10)
# 保存
model.save_model('model.pth')
# 恢复
model = MixClassifier(load_file='model.pth')
```
### 2.3 train.py    
端到端训练使用

### 2.4 predict.py
输出测试结果

### 2.5 get_features.ipynb
提取resnet、densenet、inception特征向量

### 2.6 train_predict.ipynb
使用get_features.ipynb特质向量做transfer learning。要比端到端的fine-tuning快很多。

## kaggle 分数
* resnet152 0.40934
* densenet169 0.66725
* resnet152_densenet169 features 0.40715
* inception ??

## 总结下
总体得分没有大神们的好，inception实验结果有点不正常，需要进一步研究下。



