function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% number of training examples
m = length(y); 


g=sigmoid(X*theta);

% Regularised cost function
J=1/m*(-y'*log(g)-(1-y)'*log(1-g)) + lambda/2/m* sum(theta(2:end).^2); % scaler

% Regularised partial derivatives of cost function
grad=1/m*X'*(g-y) +lambda/m *[0; theta(2:end)]; % vector size(theta)



% =============================================================

end
