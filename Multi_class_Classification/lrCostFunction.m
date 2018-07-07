function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% number of training examples
m = length(y); 

g=sigmoid(X*theta);

% Compute cost function with regularisation
% bias term not regularised by (theta(2:end)) and [0; theta(2:end)]

J=1/m*(-y'*log(g)-(1-y)'*log(1-g)) + lambda/2/m* sum(theta(2:end).^2); % scaler

% Compute partial derivatives of cost function
grad=1/m*X'*(g-y) +lambda/m *[0; theta(2:end)]; % vector size(theta)


% =============================================================
% Unroll gradient
grad = grad(:);

end
