import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import train_data_loader, test_data_loader
import warnings
warnings.simplefilter("ignore", UserWarning)
"""
Author: Yang Sizhe (A0236299N), Wang Jiangyi (A0236307J)
"""


###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""
def train(train_data_dir, model_dir, **kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    # using gpu/ cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Load pre-trained model
    model = torchvision.models.resnet152(pretrained=True)
    model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=model.fc.in_features, out_features=15), 
                                   torch.nn.Softmax())
    model.to(device)
    train_loader, val_loader, train_num, val_num = train_data_loader(train_data_dir, batch_size=32)

    # model setting
    loss_function = nn.CrossEntropyLoss()
    learning_rate = 0.01
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    best_acc = 0.0
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    epochs = 8

    # train the model
    epochs = 8

    for epoch in range(epochs):
        model.train()
        running_loss_train = 0.0
        val_acc = 0.0
        train_accurate = 0.0
        train_bar = tqdm(train_loader)
        for images, labels in train_bar:
            optimizer.zero_grad()
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            predict = torch.max(outputs, dim=1)[1]
            train_accurate += torch.eq(predict, labels.to(device)).sum().item()
            running_loss_train += loss.item()
        
        train_accurate = train_accurate / train_num
        running_loss_train = running_loss_train / train_num
        train_acc_list.append(train_accurate)
        train_loss_list.append(running_loss_train)
        print('[epoch %d] train_loss: %f  train_accuracy: %f' % (epoch + 1, running_loss_train, train_accurate))

  
        model.eval()  
        with torch.no_grad():
            val_loader = tqdm(val_loader)
            for val_images, val_labels in val_loader:
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_accurate = val_acc / val_num
            val_acc_list.append(val_accurate)

            print('[epoch %d] val_accuracy: %f' % (epoch + 1, val_accurate))
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), model_dir)
    return best_acc


    
def test(test_data_dir, model_dir, **kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    # using gpu/ cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Load the saved ResNet model
    model = torchvision.models.resnet152(pretrained=True)
    model.fc = torch.nn.Sequential(torch.nn.Linear(in_features=model.fc.in_features, out_features=15),
                                   torch.nn.Softmax())
    model.to(device)
    model.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))

    model.eval()
    test_loader, test_num = test_data_loader(test_data_dir)
    acc = 0.0  
    with torch.no_grad():
        test_loader = tqdm(test_loader)
        for test_images, test_labels in test_loader:
            outputs = model(test_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()
    test_accurate = acc / test_num
    print('test accuracy is: %f', test_accurate)
    return test_accurate


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pkl', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)






