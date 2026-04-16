function [input, target]= NARMA_P_Github(total_L, P, seed)
    
    if P < 2
        error('P must be equal to or greater than 2.')
    end   
    
    rng(seed); % Seed for reproducability 
    
    u = 0.5*rand(1,total_L);
    y = zeros(1,total_L);
    for i = P:total_L-1
        y(i+1) = 0.3*y(i) + 0.05*y(i)*sum(y(i-(P-1):i)) + 1.5*u(i-(P-1))*u(i) + 0.1;
            
        % Add a cliping functionality in case NARMA series blows off
        % usually a tanh() nonlinearity is used to keep the series from
        % diverging
        if y(i+1) > 1
            y(i+1) = mean(y(1:i)); 
        end    
    end

    input = u; 
    target = y;
end