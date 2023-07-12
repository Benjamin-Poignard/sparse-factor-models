function MSE = mse(target,Lambda)

MSE = norm(vec(target)-vec(Lambda))^2;