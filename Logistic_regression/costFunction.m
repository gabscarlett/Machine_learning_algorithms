function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% number of training examples
m = length(y); 


g=sigmoid(X*theta);

% Cost function
J=1/m*(-y'*log(g)-(1-y)'*log(1-g)); % scaler

% Partial derivatives of cost function
grad=1/m*X'*(g-y); % vector size(theta)


% =============================================================

end
