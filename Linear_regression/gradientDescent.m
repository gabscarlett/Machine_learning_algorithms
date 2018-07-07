function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% number of training examples
m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters


    h = X*theta; % update values
    theta(1) = theta(1) - alpha/m *sum((h-y).*X(:,1));
    theta(2) = theta(2) - alpha/m *sum((h-y).*X(:,2));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
