function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix withn samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%
%
%
%
% OUTPUT:
%    weights = (d+1) * 1 vector of weights where the weights correspond to
%              the columns of "data"
%              iterations to execute (useful when debugging in case your
%              code is not converging correctly!)
%              (if unspecified can be set to 1000)
%
[N, D] = size(data);  % data should have been added with one bias dim.
weights = zeros(D, 1);

for iter = 1:maxiter
    [grad] = logisitc_loss_grad(data, labels, weights);
    
    weights = weights - epsilon .* grad;
    % fprintf('iter %d, loss: %g \n', iter, loss);
end

end


function [grad] = logisitc_loss_grad(X, y, w)
N = size(X, 1);
yXw = y .* (X * w);
% a = max(max(yXw));
% loss = sum( log( exp(-a) + exp(- yXw - a) ) + a ) ./ N;

grad = - X' * (y./ (1 + exp(min(yXw, 100))) ) ./ N;

end


