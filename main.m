epsilon = 1e-5;
maxiter = 1000;
labels(labels==0) = -1;
if size(data, 1) < 58
    data = [data, ones(size(data, 1))];
end
for n = [200, 500, 800, 1000, 1500, 2000]
    [weights] = logistic_train(data(1:n, :), labels(1:n), epsilon, maxiter);
    acc = logistic(data(2001:4601, :), labels(2001:4601), weights);
    disp([num2str(n), ': ', num2str(acc)]);
end
