from torch import nn
from torchvision.models import resnet18


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class SimpleDetector(nn.Module):
    """ VGG11 inspired feature extraction layers """
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
#            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
 #           nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
  #          nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
   #         nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
    #        nn.Dropout(),
            nn.Linear(16, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        self.regressor = nn.Sequential(
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4), #4 because bounding box is defined by 4 values
            nn.Sigmoid()
        )

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        return self.classifier(x), self.regressor(x)


class DeeperDetector(nn.Module):
    """ VGG11 inspired feature extraction layers """
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/2^5 = 7
            nn.Linear(512 * 7 * 7, 32),
            nn.ReLU(),
   #         nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
    #        nn.Dropout(),
            nn.Linear(16, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        self.regressor = nn.Sequential(
            nn.Linear(512*7*7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4), #4 because bounding box is defined by 4 values
            nn.Sigmoid()
        )
        self.regressor.apply(init_weights)

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        return self.classifier(x), self.regressor(x)

class VGGLikeDetector(nn.Module):
    """ VGG11 inspired feature extraction layers """
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(512*14*14, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        self.regressor = nn.Sequential(
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4), #4 because bounding box is defined by 4 values
            nn.Sigmoid()
        )

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        return self.classifier(x), self.regressor(x)
    
class ResnetObjectDetector(nn.Module):
    """ Resnet18 based feature extraction layers """
    def __init__(self, nb_classes):
        super().__init__()
        # copy resnet up to the last conv layer prior to fc layers, and flatten
        features = list(resnet18(pretrained=True).children())[:9]
        self.features = nn.Sequential(*features, nn.Flatten())

        for param in self.features.parameters():
            param.requires_grad = False

        # create classifier path for class label prediction
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
     #       nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
      #      nn.Dropout(),
            nn.Linear(512, nb_classes)
        )

        # create regressor path for bounding box coordinates prediction
        self.regressor = nn.Sequential(
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4), #4 because bounding box is defined by 4 values
            nn.Sigmoid()
        )

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        x = self.features(x)
        return self.classifier(x), self.regressor(x)
