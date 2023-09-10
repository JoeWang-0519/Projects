from torchvision.transforms import transforms
import random
import torch
import torchvision

dist = {'bedroom': 1, 'Coast': 2, 'Forest': 3,
        'Highway': 4, 'industrial': 5, 'Insidecity': 6,
        'kitchen': 7, 'livingroom': 8, 'Mountain': 9,
        'Office': 10, 'OpenCountry': 11, 'store': 12,
        'Street': 13, 'Suburb': 14, 'TallBuilding': 0}
label_class = ['bedroom', 'Coast', 'Forest',
               'Highway', 'industrial', 'Insidecity',
               'kitchen', 'livingroom', 'Mountain',
               'Office', 'OpenCountry', 'store',
               'Street', 'Suburb', 'TallBuilding']

data_transform = {"train": transforms.Compose([
                           transforms.Resize((224, 224)),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
 
                  "val": transforms.Compose([
                         transforms.Resize((224, 224)),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])}

def train_data_loader(dir='./data/train/', batch_size=32):

    # fix random seed = 5203
    seed = 5203
    random.seed(seed)
    torch.manual_seed(seed)
    ori_dataset = torchvision.datasets.ImageFolder(root=dir, transform=None)
    dataset = torchvision.datasets.ImageFolder(root=dir, transform=None,
                                               target_transform=lambda x: dist[ori_dataset.classes[x]])
    dataset.class_to_idx = dist
    dataset.classes = label_class

    # 5% validation; 95% training
    train_size = int(0.95 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = data_transform["train"]
    val_dataset.dataset.transform = data_transform["val"]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, len(train_dataset), len(val_dataset)

def test_data_loader(dir='./data/test/', batch_size=32):

    dataset = torchvision.datasets.ImageFolder(root=dir, transform=None)
    test_dataset = torchvision.datasets.ImageFolder(root=dir, transform=None,
                                                    target_transform=lambda x: dist[dataset.classes[x]])
    test_dataset.class_to_idx = dist
    test_dataset.classes = label_class

    test_dataset.transform = data_transform["val"]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, len(test_dataset)
