function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% number of training examples
m = length(y); 
J_history = zeros(num_iters, 1);

temp=zeros(size(theta));

for iter = 1:num_iters


    for i=1:size(X,2)
    temp(i) = theta(i) - (alpha/m *(X*theta-y)'*X(:,i))';
    
    end
    
    % UPDATE
    theta=temp;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
