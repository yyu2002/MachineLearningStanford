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
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

K = num_labels;
Y = eye(K)(y, :);

% Part 1: Foreward Feed and Cost Function

a1 = [ones(m, 1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1), a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

cost = sum((-Y .* log(a3)) - ((1 - Y) .* log(1 - a3)), 2);
J = (1 / m) * sum(cost);

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);

reg = (lambda / (2 * m)) * (sum(sumsq(Theta1NoBias)) + sum(sumsq(Theta2NoBias)));
J += reg;

% Part 2: Backpropagation

Delta1 = 0;
Delta2 = 0;

for t = 1:m
	% input values
	a1 = [1; X(t, :)'];
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	% delta output layer
	d3 = a3 - Y(t, :)';

	% delta hidden layer
	d2 = (Theta2NoBias' * d3) .* sigmoidGradient(z2);

	% accumulate
	Delta2 += d3 * a2';
	Delta1 += d2 * a1';
endfor

% normal gradient
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

% Part 3: regularized gradient
Theta1_grad(:, 2:end) += ((lambda / m) * Theta1NoBias);
Theta2_grad(:, 2:end) += ((lambda / m) * Theta2NoBias);

% unroll grandients
grad = [Theta1_grad(:); Theta2_grad(:)];

end
