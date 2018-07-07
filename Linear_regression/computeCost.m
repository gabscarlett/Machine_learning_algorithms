function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% number of training examples
m = length(y); 


% Hypothesis
h=X*theta;
% Cost function
J=1/2/m *sum((h-y).^2);


end
