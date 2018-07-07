function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad is an "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);



% ====================== ALGORITHM ======================


% Part 1: Feedforward the neural network and return the cost in the
%         variable J.

% recode y to Y
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end


% Compute hidden layer a2 matrix
a1= [ones(m, 1) X];
z2 = a1 * Theta1';
a2= sigmoid(z2);

% Add ones to the hidden layer a2 matrix
a2=[ones(m,1) a2];

% Compute output layer a3
z3=a2 * Theta2';
a3=sigmoid(z3);


% Compute cost function
J=-1/m*sum(sum(Y.*log(a3)+(1-Y).*log(1-a3)));

% Add regularisation to cost function
% NOTE: bias term not regularised by (theta(2:end)) and [0; theta(2:end)]
T1=Theta1(:,2:end);
T2=Theta2(:,2:end);
reg= lambda/2/m*sum([T1(:); T2(:)].^2);
J=J+reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 


    
% Calculate output layer delta value
delta3=a3-Y;

% Calculate cderivative of activation function
gPrime2 = sigmoidGradient(z2);
gPrime2=[ones(m,1) gPrime2];

% Calculate hidden layer delta value    
% NOTE: assuming only one hidden layer
delta2=delta3*Theta2.*gPrime2;    
delta2=delta2(:,2:end);
    
% Accumalate    
Del_1 = delta2'*a1;
Del_2= delta3'*a2;
    
% Partial derivative of J
Theta1_grad = 1/m*Del_1;
Theta2_grad = 1/m*Del_2;   
      
    
    
% Part 3: Implement regularization with the cost function and gradients.


% Partial derivative of J with regularisation
% NOTE: bias term not regularised by Theta1(:,1)=0; and Theta2(:,1)=0;
Theta1(:,1)=0; 
Theta1_grad = 1/m*(Del_1 +lambda*Theta1);
Theta2(:,1)=0;
Theta2_grad = 1/m*(Del_2 +lambda*Theta2); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
