from torchvision.transforms import transforms
from kaggleImageFolder import KaggleImageFolder
import torch


def read_data(data_dir, mode, labels_csv='labels.csv', split_p=0.2,
              mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if mode == 'train':
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    if mode == 'test':
        return KaggleImageFolder(data_dir, transform=data_transform, mode='test')
    else:
        return KaggleImageFolder(data_dir, labels_csv, mode=mode, transform=data_transform, split_p=split_p)


def accuracy(model, dataloader, n_data, gpu=torch.cuda.is_available()):
    correct = 0
    model.eval()
    for x, y in dataloader:
        if gpu:
            x = x.cuda()
            y = y.cuda()
        z = model(x)
        yhat = torch.argmax(z, 1)
        correct += (y == yhat).sum().item()

    acc = correct / n_data
    return acc

