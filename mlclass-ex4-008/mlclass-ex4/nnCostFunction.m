function [J grad] = nnCostFunction(nn_params, ...
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
%   The returned parameter grad should be a "unrolled" vector of the
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m, 1) X];         %5000x401
a2 = sigmoid(X*Theta1');    %5000x25
a2 = [ones(m,1) a2];        %5000x26
a3 = sigmoid(a2*Theta2');   %5000x10
%a3 = [ones(m,1) a3];
Y = zeros(m,num_labels);    %5000x10
for (i = 1:num_labels:m)
    Q = eye(num_labels);
    Y(i:(i+num_labels-1),1:num_labels) = Q(y(i:(i+num_labels-1)),:);
end

j3 = 0;
for i = 1:m
   j2 = 0;
    for k =1:num_labels
       j1 = -Y(i,k)*log(a3(i,k))-(1-Y(i,k))*log(1-a3(i,k));
       j2 = j2+j1;
    end
    j3 = j3 + j2;
end

J = j3/m;
t1 = Theta1(:,2:end);
t2 = Theta2(:,2:end);
J = J+lambda*0.5*(sum(sum(t1.*t1))+sum(sum(t2.*t2)))/m;

z2 = X*Theta1';     %5000x25
%z2 = [ones(m,1) z2];%5000x26
delta3 = a3 - Y;    %5000x10
delta2 = (delta3*(Theta2(:,2:end))).*sigmoidGradient(z2);%5000x25
Theta1_grad = (Theta1_grad + delta2'*X)/m;%25x401
Theta2_grad = (Theta2_grad + delta3'*a2)/m;%10x26
ee = Theta1(:,2:end);
ee = [zeros(hidden_layer_size,1) ee];
gg = Theta2(:,2:end);
gg = [zeros(num_labels,1) gg];
Theta1_grad = Theta1_grad +lambda*ee/m;
Theta2_grad = Theta2_grad +lambda*gg/m;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
