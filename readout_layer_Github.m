function [P_drop, X_drop, time_axis_reduced, X_train, X_test, y_train, y_test, y_train_hat, y_test_hat] = readout_layer_Github(E_drop, target, time_axis, steps_per_bit, N_mask, Warm1, Warm2, Ltrain, Ltest, total_L)

    % Make sure that the number of time steps per bit can be evenly divided
    % into N_mask virtual nodes.
    assert(mod(steps_per_bit, N_mask) == 0, 'steps_per_bit must be divisible by N_mask');

    % Number of solver steps per virtual node.
    M = steps_per_bit / N_mask;

    % Convert the optical field at the drop port into optical intensity.
    P_drop = abs(E_drop).^2;

    % Make sure that the total number of power samples can be evenly grouped
    % into segments of length M.
    assert(mod(numel(P_drop), M) == 0, 'numel(P_drop) must be divisible by M');

    % Make sure that the total number of time samples can be evenly grouped
    % into segments of length M.
    assert(mod(numel(time_axis), M) == 0, 'numel(time_axis) must be divisible by M');

    % Reduce the original time axis by averaging over M samples.
    time_axis_reduced = mean(reshape(time_axis, M, []), 1);

    % Reduce the vector size of drop-port intensity by averaging over M samples.
    X_drop = mean(reshape(P_drop, M, []), 1);

    % Reshape the averaged optical intensity values into a matrix of size
    % [total number of bits] x [number of virtual nodes].
    X_drop = reshape(X_drop, N_mask, total_L).'; 

    % Extract the training feature matrix after removing the initial warm-up.
    X_train = X_drop(Warm1+1:Warm1+Ltrain,:); 

    % Extract the testing feature matrix after the training set and the
    % second warm-up interval.
    X_test = X_drop(Warm1+Ltrain+Warm2+1:end,:);
 
    % Extract the training target values corresponding to X_train.
    y_train = target(Warm1+1 : Warm1+Ltrain);

    % Make sure the training target is a column vector.
    y_train = y_train(:);

    % Extract the testing target values corresponding to X_test.
    y_test = target(Warm1+Ltrain+Warm2+1 : Warm1+Ltrain+Warm2+Ltest);

    % Make sure the testing target is a column vector.
    y_test = y_test(:);

    % Regularization parameter used for ridge regression.
    LAMBDA = 1e-9; % FDRO: was this optimized?

    % Train the linear readout weights using ridge regression.
    % The first element of W is the bias term.
    W = ridge(y_train, X_train, LAMBDA, 0);

    % Compute the predicted training output using the trained readout weights.
    y_train_hat = W(1) + X_train * W(2:end);

    % Compute the predicted testing output using the trained readout weights.
    y_test_hat = W(1) + X_test * W(2:end);

    % Replace the final predicted test value with the previous one, because
    % the last predicted value is quite off. 
    y_test_hat(end) = y_test_hat(end-1); 

end
