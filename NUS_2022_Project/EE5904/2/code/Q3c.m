clear all;
clc;

load datasample.mat

% since the activation function is logistic function, we set the target as follows:
% 0.2->0 (manmade objects)
% 0.8->1 (dog)

% target is the 0-1 target
target = target_all;
% target all is the 0.2-0.8 target
target_all = [0.2*ones(1,500), 0.8*ones(1,500)];


% split the training and validation set
train_image = image_all(:, [1:450, 501:950]);
val_image = image_all(:, [451:500, 951:1000]);

train_target = target_all([1:450, 501:950]);
val_target = target_all([451:500, 951:1000]);

t_target = target([1:450, 501:950]);
v_target = target([451:500, 951:1000]);

% we apply the normalization on the row vector
mu1 = mean(train_image, 2);
s1 = std(train_image, 0, 2);
for i = 1:1024
    train_image_norm1(i,:) = (train_image(i,:) - mu1(i)) / s1(i);
    val_image_norm1(i,:) = (val_image(i,:) - mu1(i)) / s1(i);
end

%
R_val = [];
R_train = [];
P_val = [];
P_train = [];
A_val = [];
A_train = [];

% hidden neuron number (55 different number)
h = [300];
idx = 1;
for num = h

    net = patternnet(num);
    % for pattern recognition, we try 'trainrp'
    net.trainFcn = 'trainrp';
    net.trainParam.epochs = 30;
    %net.divideFcn = 'dividetrain';
    net.performParam.regularization = 0.999;
    net.trainParam.max_fail = 30;


    net = train(net, train_image_norm1, train_target);
    
    pred_train = round(net(train_image_norm1));
    pred_val = round(net(val_image_norm1));
    
    [recall_train, precision_train, accu_train] = perf_meas(pred_train, t_target);
    R_train(idx) = recall_train;
    P_train(idx) = precision_train;
    A_train(idx) = accu_train;
    
    % we care about this part
    [recall_val, precision_val, accu_val] = perf_meas(pred_val, v_target);
    R_val(idx) = recall_val;
    P_val(idx) = precision_val;
    A_val(idx) = accu_val;

    idx = idx + 1;
end


figure(1);

xaxis = 1:55;

subplot(3,1,1);
plot(xaxis, R_val, '-*r');
subtitle('Recall');
set(gca,'xtick', [1, 5:5:55]);
set(gca,'xticklabel', [5, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 800]);
xlabel('number of hidden neurons');
ylabel('Recall');
grid on;
%title('early stop with regularization r=0.25');

subplot(3,1,2);
plot(xaxis, P_val, '--db');
subtitle('Precision');
set(gca,'xtick', [1, 5:5:55]);
set(gca,'xticklabel', [5, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 800]);
xlabel('number of hidden neurons');
ylabel('Precision');
grid on;

subplot(3,1,3);
plot(xaxis, A_val, '--^k');
hold on;
plot(xaxis, A_train, 'm')
subtitle('Accuracy');
set(gca,'xtick', [1, 5:5:55]);
set(gca,'xticklabel', [5, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 800]);
xlabel('number of hidden neurons');
ylabel('Accuracy');
legend('testing accuracy', 'training accuracy');
grid on;

%we set hidden neurons = 175 first, then try to use SVD to find the
%appropriate number of hidden neurons




