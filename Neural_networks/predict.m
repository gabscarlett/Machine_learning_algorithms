function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% number of training examples
m = size(X, 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


% Compute hidden layer A2 matrix
A2 = sigmoid(X * Theta1');

% Add ones to the hidden layer A2 matrix
A2=[ones(m,1) A2];

% Compute output layer A3
A3=sigmoid(A2 * Theta2');

h=A3; % Hypothesis function

%Predict
[~, p]=max(h,[],2); % determines the column K for the maximum element in each row

% =========================================================================


end
