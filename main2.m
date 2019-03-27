load('ad_data.mat');
% Specify the options (use without modification).
opts.rFlag = 1;  % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4;  % termination options.
opts.maxIter = 5000; % maximum iterations.

for par = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    if par > 0
        [w, c] = LogisticR(X_train, y_train, par, opts);
    else
        w = logistic_train([X_train, ones(size(X_train, 1), 1)], y_train, 1e-3, 5000);
        c = w(size(w, 1));
        w = w(1:size(w, 1)-1);
    end
    scores = X_test * w + c;
    y = y_test > 0;
    [~, ~, ~, auc] = perfcurve(y_test, scores, 1);
    fprintf('par: %g, auc: %g, #feature: %d\n', par, auc, sum(abs(w)>1e-12, 1));
end
