% Question 2 MNIST loading the dataset
clear all;
clc;
load mnist_m.mat;
% train_data: training data, 784x1000 matrix
% train_classlabel: the labels of the training data, 1x1000 vector
% test_data: test data, 784x250 matrix
% train_classlabel: the labels of the test data, 1x250 vector

% visualization
col = 420;
tmp=reshape(train_data(:,col),28,28);
imshow(tmp);
train_classlabel(col),

% since student ID is: A0236307J, then we should focus on 0 and 7

% find the location of classes 0, 7
% train_idx07 = find(train_classlabel==0 | train_classlabel==7); 
% test_idx07 = find(test_classlabel==0 | test_classlabel==7); 
% 
% % Re-label
% train_target = zeros(1, 1000);
% test_target = zeros(1, 250);
% train_target(train_idx07) = 1;
% test_target(test_idx07) = 1;
% 
% save mnist_modify.mat



 

