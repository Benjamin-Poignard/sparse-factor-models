function MSE = mse(Lambda_true,Lambda)

MSE = norm(vec(Lambda_true)-vec(Lambda))^2;