import torch.nn as nn
from torchvision import models
import torch

from collections import OrderedDict


class AngClassifier:
    def __init__(self, arch='vgg16', hidden_units=[1000, 500], n_class=10, load_file=None):
        self.__support_arches__ = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
                                   'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
                                   'resnet18', 'resnet34', 'resnet50', 'resnet101',
                                   'resnet152', 'densenet121', 'densenet169', 'densenet201',
                                   'densenet161', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3']

        if load_file is not None:
            checkpoint = torch.load(load_file)
            self.arch = checkpoint['arch']
            self.hidden_units = checkpoint['hidden_units']
            self.n_class = checkpoint['n_class']
            self.model = self.__create_model__()
            self.model.load_state_dict(checkpoint['param_state_dict'])
        else:
            if arch not in self.__support_arches__:
                raise Exception("arch must be one of {}, but input is {}".format(
                    self.__support_arches__,
                    arch
                ))

            self.arch = arch
            self.hidden_units = hidden_units
            self.n_class = n_class

            self.model = self.__create_model__()
            self.train = self.model.train
            self.eval = self.model.eval
            self.cuda = self.model.cuda

    def __repr__(self):
        info = "One {} breeds classifier based on {}".format(self.n_class, self.arch)
        return info

    def __call__(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def __make_classifier__(self, first_in_features, head_drop=False):
        layers = OrderedDict()
        if head_drop:
            layers['drop-h'] = nn.Dropout(0.5)

        layers['linear0'] = nn.Linear(first_in_features, self.hidden_units[0])
        layers['drop0'] = nn.Dropout(p=0.5)
        layers['relu0'] = nn.ReLU(inplace=True)

        for i, (input_dim, output_dim) in enumerate(zip(self.hidden_units[0:], self.hidden_units[1:]), 1):
            layers['linear' + str(i)] = nn.Linear(input_dim, output_dim)
            layers['drop' + str(i)] = nn.Dropout(p=0.5)
            layers['relu' + str(i)] = nn.ReLU(inplace=True)

        layers['linear' + str(len(self.hidden_units))] = nn.Linear(self.hidden_units[-1], self.n_class)

        return nn.Sequential(layers)

    def __create_model__(self):
        if self.arch == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif self.arch == 'vgg11':
            model = models.vgg11(pretrained=True)
        elif self.arch == 'vgg11_bn':
            model = models.vgg11_bn(pretrained=True)
        elif self.arch == 'vgg13':
            model = models.vgg13(pretrained=True)
        elif self.arch == 'vgg13_bn':
            model = models.vgg13_bn(pretrained=True)
        elif self.arch == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif self.arch == 'vgg16_bn':
            model = models.vgg16_bn(pretrained=True)
        elif self.arch == 'vgg19_bn':
            model = models.vgg19_bn(pretrained=True)
        elif self.arch == 'vgg19':
            model = models.vgg19(pretrained=True)
        elif self.arch == 'resnet18':
            model = models.resnet18(pretrained=True)
        elif self.arch == 'resnet34':
            model = models.resnet34(pretrained=True)
        elif self.arch == 'resnet50':
            model = models.resnet50(pretrained=True)
        elif self.arch == 'resnet101':
            model = models.resnet101(pretrained=True)
        elif self.arch == 'resnet152':
            model = models.resnet152(pretrained=True)
        elif self.arch == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif self.arch == 'densenet169':
            model = models.densenet169(pretrained=True)
        elif self.arch == 'densenet201':
            model = models.densenet201(pretrained=True)
        elif self.arch == 'densenet161':
            model = models.densenet161(pretrained=True)
        elif self.arch == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=True)
        elif self.arch == 'squeezenet1_1':
            model = models.squeezenet1_1(pretrained=True)
        elif self.arch == 'inception_v3':
            model = models.inception_v3(pretrained=True)
        else:
            raise Exception("arch must be one of {}, but input is {}".format(
                self.__support_arches__,
                self.arch
            ))

        for param in model.parameters():
            param.requires_grad = False

        if 'alexnet' in self.arch:
            first_in_features = model.classifier[1].in_features
            model.classifier = self.__make_classifier__(first_in_features, True)
            self.classifier = model.classifier
        elif 'vgg' in self.arch:
            first_in_features = model.classifier[0].in_features
            model.classifier = self.__make_classifier__(first_in_features)
            self.classifier = model.classifier
        elif 'resnet' in self.arch:
            first_in_features = model.fc.in_features
            model.fc = self.__make_classifier__(first_in_features)
            self.classifier = model.fc
        elif 'densenet' in self.arch:
            first_in_features = model.classifier.in_features
            model.classifier = self.__make_classifier__(first_in_features)
            self.classifier = model.classifier
        elif 'squeezenet' in self.arch:
            final_conv = nn.Conv2d(512, self.n_class, kernel_size=1)
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            model.num_classes = self.n_class
            self.classifier = model.classifier
        elif 'inception' in self.arch:
            model.aux_logits = False
            first_in_features = model.fc.in_features
            model.fc = self.__make_classifier__(first_in_features)
            self.classifier = model.fc

        return model

    def save_model(self, save_path):
        checkpoint = {
            'arch': self.arch,
            'param_state_dict': self.model.state_dict(),
            'hidden_units': self.hidden_units,
            'n_classes': self.n_class
        }
        torch.save(checkpoint, save_path)


if __name__ == '__main__':
    arches = ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
              'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
              'resnet18', 'resnet34', 'resnet50', 'resnet101',
              'resnet152', 'densenet121', 'densenet169', 'densenet201',
              'densenet161', 'squeezenet1_0', 'squeezenet1_1', 'inception_v3']

    for arch in arches:
        test_model = AngClassifier(arch)
        print(test_model)




