function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

sig = sigmoid(X * theta);
Jreg = lambda / (2*m) * (theta' * theta - theta(1) .^2);

J = (1 / m) * (-y' * log(sig) - (1-y)' * log(1 - sig)) + Jreg;

temp = theta;
temp(1) = 0;
gradReg = (lambda / m) * temp;

grad = (1 / m) * X' * (sig - y) + gradReg;

grad = grad(:);

end
