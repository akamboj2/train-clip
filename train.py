import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.text_image_dm import TextImageDataModule
from models import CLIPWrapper

import os
from torchvision import transforms
import torch
import torchvision
from pytorch_lightning import LightningDataModule
import pytorch_lightning
import clip
import random

import tensorboard


global_batch_size = 8

template = "A person is "
gt_labels = ["Cook.Cut"             ,
            "Cook.Usemicrowave"     ,
            "Cook.Useoven"          ,
            "Cook.Usestove"         ,
            "Drink.Frombottle"      ,
            "Drink.Fromcup"         ,
            "Eat.Snack"             ,
            "Eat.Useutensil"        ,
            "Exercise"              ,
            "Getup"                 ,
            "Lay.Onbed"             ,
            "Nap"                   ,
            "Play.Boardgame"        ,
            "Read"                  ,
            "Use.Coffeemachine"     ,
            "Use.Computer"          ,
            "Use.Dishwasher"        ,
            "Use.Gamecontroller"    ,
            "Use.Kettle"            ,
            "Use.Mop"               ,
            "Use.Phone"             ,
            "Use.Refrig"            ,
            "Use.Shelf"             ,
            "Use.Sink"              ,
            "Use.Switch"            ,
            "Use.Tablet"            ,
            "Use.Vaccum"            ,
            "Watch.TV"              ,
            "Write"                             
            ]
sentences = ["cooking by cutting something",
    "cooking using a microwave",
    "cooking using an oven",
    "cooking using a stove",
    "drinking from a bottle",
    "drinking from a cup",
    "eating a snack",
    "eating using a utensil",
    "exercising",
    "getting up",
    "laying on a bed",
    "napping",
    "playing a boardgame",
    "reading",
    "using a coffee machine",
    "using a computer",
    "using a dishwasher",
    "using a gname controller",
    "using a kettle",
    "using a mop",
    "using a phone",
    "using a refrigerator",
    "using a shelf",
    "using a sink",
    "using a ninetendo switch",
    "using a tablet",
    "using a vaccum",
    "watching TV",
    "writing"
]
text_labels = [template + w for w in sentences]
text_dict = {a:b for a,b in zip(gt_labels,text_labels)}

#for debugging:
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]

        #use class text as label, not index
        class_txts = text_dict[self.classes[original_tuple[1]]]
        #self.classes translates the class indices to words (use.shelf, etc.)
        #text dict makes the labels full sentences

        tokenized_text =  clip.tokenize(class_txts)[0]

        # make a new tuple that includes original and the path
        # tuple_with_path = ((original_tuple[0],) + (tokenized_text,) + (path,))
        # return tuple_with_path

        #taking out path for pytorch lightning
        return  original_tuple[0], tokenized_text

#function to load dataset
def load_dataset():
    traindir = os.path.join('/home/abhi/research/SmartHome/Data/image_folder_data', 'train')
    valdir = os.path.join('/home/abhi/research/SmartHome/Data/image_folder_data', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),

        # util.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
        # normalize,
    ])

    train_dataset = ImageFolderWithPaths( #for debuggging
        traindir,
        transform=train_transforms
    )
    train_dataset_small = train_dataset
    train_dataset_small = torch.utils.data.Subset(
        train_dataset_small, 
        random.sample(range(len(train_dataset)), k=int(len(train_dataset)/20)))

    trainloader = torch.utils.data.DataLoader(
        train_dataset_small, batch_size=global_batch_size, shuffle=True,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=True,
        # sampler=torch.utils.data.distributed.DistributedSampler(train_dataset_small, shuffle=True)
    )

    val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
            # normalize
        ])
    val_dataset = ImageFolderWithPaths( #for debuggging
        valdir,
        transform=val_transforms
    )
    
    val_dataset_small = val_dataset
    val_dataset_small = torch.utils.data.Subset(
        val_dataset_small,
        random.sample(range(len(val_dataset)), k=int(len(val_dataset)/10)))

    valloader = torch.utils.data.DataLoader(
        val_dataset_small,
        batch_size=global_batch_size, shuffle=False,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=False
    )

    return trainloader, valloader, train_dataset_small

class TextImageDataModule(LightningDataModule):
    def __init__(self,
                 ):
        super().__init__()

    def setup(self, stage=None):
        self.dataloader_train, self.dataloader_test,self.dataset = load_dataset()
        return self.dataset
    def prepare_data(self):
        pass
    def train_dataloader(self):
        # sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=True)
        # dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, sampler=sampler, shuffle=False)
        # return dataloader
        return self.dataloader_train
    def val_dataloader(self):
        return self.dataloader_test


   
def main(hparams):
    config_dir = 'models/configs/ViT.yaml' if 'ViT' in hparams.model_name else 'models/configs/RN.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)[hparams.model_name]

    # if hparams.minibatch_size < 1:
    #     hparams.minibatch_size = hparams.batch_size

    model = CLIPWrapper(hparams.model_name, config, hparams.minibatch_size)
    del hparams.model_name
    # dm = TextImageDataModule.from_argparse_args(hparams)
    trainer = Trainer.from_argparse_args(hparams, precision=16, max_epochs=32, devices=1, \
        accelerator="gpu",strategy="ddp")
    # trainer.fit(model, dm)

    
    dm = TextImageDataModule()
    trainer.fit(model,dm)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--minibatch_size', type=int, default=0)
    parser = TextImageDataModule.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)

    # dataloader_train, dataloader_test = load_dataset()
    # for ind, (images,labels,paths) in enumerate(dataloader_train):
    #     for i in range(32):
    #         print(labels[i])
    #         print(paths[i])
    #         print()
    #     break
    # print(labels,paths)
    # for i,j in text_dict.items():
    #     print(i,j)

