
%% Data loading and preprocessing
% Load Data
df = readtable('./data/Crop_recommendation.csv');

% Split data into features and labels
features = df{:, 1:end-1};
labels = df{:, end};

% Convert labels to categorical
labels = categorical(labels);

% Normalize features
meanVal = mean(features);
stdVal = std(features);
features = (features - meanVal) ./ stdVal;

% Save the normalization parameters
save('./model/normalization/normalization.mat', 'meanVal', 'stdVal');

% Split data into training and validation sets
cv = cvpartition(size(features, 1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxValidation = test(cv);

featuresTrain = features(idxTrain, :);
labelsTrain = labels(idxTrain);
featuresValidation = features(idxValidation, :);
labelsValidation = labels(idxValidation);

%% Define the NN
layers = [
    featureInputLayer(7) % Input layer size based on feature dimensions
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(128)
    reluLayer
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(22) % Number of classes
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',100, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{featuresValidation, labelsValidation}, ...
    'Plots','training-progress', ...
    'Verbose',false);

%% Train the Network
net = trainNetwork(featuresTrain, labelsTrain, layers, options);
% Save the trained network
modelPath = './model/trainedModel.mat';
save(modelPath, 'net');
%% Evaluate the Network
predictedLabels = classify(net, featuresTrain);
accuracy = sum(predictedLabels == labelsTrain) / numel(labelsTrain);
fprintf('Accuracy of the network on the train data: %.2f%%\n', accuracy * 100);

%% Loaind the model for further use

% % Load the trained network
% modelPath = './model/trainedModel.mat';
% loadedStruct = load(modelPath);
% net = loadedStruct.net;
% 
% % Now `net` can be used for prediction
% % For example, to predict labels of the validation set
% predictedLabels = classify(net, featuresValidation);
% 
% % Calculate accuracy if true labels are known
% accuracy = sum(predictedLabels == labelsValidation) / numel(labelsValidation);
% fprintf('Accuracy of the model on validation data: %.2f%%\n', accuracy * 100);


