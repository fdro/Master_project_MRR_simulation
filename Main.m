clear all; 
clc; 

% This main script will serve as a simulation for a photonic TDRC system. 
% The script will simulate: 
% -Input layer. 
% -Reservoir layer
% -Output layer

% The input layer will consist of: 
% -Generation of data sequence, u(n)
% -Generation of mask, m(n). 
% -Generation of X(n) = m(n)*u(n)+bias
% -Simulation of ideal laser with pump frequency, wp/(2pi) and pump Power
% Pin¨
% -Simulation of Mach Zender modulator. We assume MZM works in the linear
% region, making a simplified expression for the modulated field.

% The Reservoir layer will consist of: 
% - Nonlinear description of MRR. The resonance frequency, omega_r, will depend
% on free carrier and thermal effects. 
% Solver for the three coupled differential equations describing MRR. These
% three equations will be. 
%        - da/dt = (.....), with a: complex modal amplitude 
%        - dN/dt = (.....), with N: Free carrier density 
%        - dT/dt = (.....), with T: Change in cavity temperature
% We will create our own RK4-solver, with fixed solver step-size. In order
% to make sure we get a stable and precise solver, we will perform
% normalization of the three equations. 
% - Delay loop, serving as memory component for the reservoir layer. 

% The output layer will consist of: 
% Photo-diode, measuring intensity of light. 
% Weight training, with the use of Ridge regression
% Prediction and validation 

% The script will also contain all parameter values used in the simulation



%% Parameter values. Mainly gathered from Bernards PhD. 
nsi = 3.485;                                            % Unpertubated refractive index for Silicon [~]
cp = 0.7e3;                                             % specific heat capacity at constant pressure [J*(kg*K)^(-1)]
m = 1.2e-14;                                            % mass MRR [kg]
L = 2*pi*7.5e-6;                                        % Circumerence MRR [m]
B_TPA = 8.4e-12;                                        % TPA coefficient [m*W^(-1)]
alpha_dB = 0.8*1e2;                                     % Attenuation per meter in desibel [dB/m]
alpha_lin = alpha_dB*log(10)/10;                        % Attenuation per meter linear [1/m]
zeta = 0.95;                                            % Coupling coefficient over to delay loop  

dnsi_dT = 1.86e-4;                                      % d(nsi)/dT [K^(-1)]
dnsi_dN = -1.73e-27;                                    % d(nsi)/dN [m^(3)]

GAMMA_TH = 0.9355;                                      % Confinement factor for temperature [~]
GAMMA_FCA = 0.9996;                                     % FCA confinement factor [~]
GAMMA_TPA = 0.9964;                                     % TPA confinement factor [~~]
sigma_fca = 1e-21;                                      % FCA coefficient in Silicon [m^2]
V_FCA = 2.36e-18;                                       % FCA effective mode volume [m^3]
V_TPA = 2.59e-18;                                       % TPA effective mode volume [m^3]

c = 299792458;                                          % Speed of light [m/s]
h = 6.62607015e-34 ;                                    % Plancks constant [J*s]
h_bar = h/2/pi;                                         % Reduced Plancks constant [J*s]
lambda_resonance_0 = 1552.89e-9;                        % Resonance wavelength in cold cavity [m]
omega_resonance_0 = 2*pi*c/lambda_resonance_0;          % Resonance angular frequency in cold cavity [rad/s]


% Time constants
tau_c = 54.7e-12;                                       % Coupling lifetime [s]
tau_r = nsi/alpha_lin/c;                                % Internal loss lifetime

tau_th = 50e-9;                                         % Thermal time coefficient
tau_fc = 10e-9;                                         % Free carrier time coefficent

t_sq = 0.9604;                                          % Self-coupling power coeffifient (t^2)
k_sq = 1-t_sq;                                          % Cross-couplung power coefficient. (k^2). Remember: t_sq + k_sq = 1
eta_lin = 0.4;                                          % Fraction of linear loss due to absorbtion. 


%% 1) Generation of data sequence

Ltrain = 2000;                                          % Length of Training set
Ltest = 500;                                            % Length of Testing set
Warm1 = 100;                                            % Length of warm-up. Helps to remove initial transients such that the MRR is in a steady state when processing the training data.
Warm2 = 300;                                            % Sits between training- and test set. Helps to remove any memory of the training data in the MRR when we perform prediction on the test data.
                                                        
total_L   = Warm1 + Ltrain + Warm2 + Ltest;             % Total number of bits for warmup, training, seperation, testing. 




% Create mask and set number of time steps per bit. 
% Since we work with a discrete time grid, it is important that both N_mask_raw, steps_per_bit_raw and
% (steps_per_bit_raw / N_mask_raw) are integers. If (steps_per_bit_raw /
% N_mask_raw) is not an integer, the function "change_N_mask" will alter the
% values of
% either N_mask_raw, steps_per_bit or both in order to fulfill the demand.
N_mask_raw = 50;                                         % Number of wanted virtal nodes.
steps_per_bit_raw = 1000;                                 % Time steps per bit. Must be an integer.
[N_mask, steps_per_bit] = change_N_mask_Github(steps_per_bit_raw, N_mask_raw); 

% Print message if N_mask or steps_per_bit have been altered. 
if N_mask == 1
   disp(['steps_per_bit / N_mask needs to be an integer. Since this is not the case, N_mask is set to N_mask = 1. The data is therefore not masked.' ...
       ' Change steps_per_bit.'])
elseif N_mask ~= N_mask_raw || steps_per_bit ~= steps_per_bit_raw
    disp(['steps_per_bit / N_mask_raw is not an integer. Therefore, N_mask is set to ', num2str(N_mask) ', and steps_per_bit is set to ', num2str(steps_per_bit) ' number of steps per bit.'])
end    

bitlength = 1e-9 * 2;                                   % Bit width [s]
solver_steps = bitlength/steps_per_bit;                 % Solver steps in RK4 methods [s]
time_window = total_L*bitlength;                        % Total time window for our data sequence [s]
time_axis = 0:solver_steps:(time_window-solver_steps);  % Time axis [s]


%% Parameters typical to adjust
% Under are the most common parameters to adjust
P = 10;                                                 % Order of NARMA_P. P must be >= 2. 
bias = 8;                                               % Bias to approximate constant light intensity into MRR.
type = "NARMA_P";                                       % Choose between "ones", "NARMA_P" 
Pin_avg_dBm = 10;                                       % Average power of optical pump, dBm
Pin_avg_lin = 10^(Pin_avg_dBm/10)*1e-3;                 % Average power of optical pump, linear scale
delta_freq = -45.0*1e9;                                 % Detuning frequency [Hz]
omega_p = omega_resonance_0 - 2*pi*delta_freq;          % Angular pump frequency [rad/s]
tau_d = 0.5e-9;                                         % Delay time in delay loop 
delta_phi = 0;                                          % Additional phase shift in delay loop
phi_d = omega_p * tau_d + delta_phi;                    % Total phase shift in delay loop 
seed = 11;                                              % Seed used for creating input signal, and target.

% Select if the simulation should be all-pass or add-drop configuraton
mode = "add_drop";                                      % Add-drop configuration
%mode = "all_pass";                                     % All-pass %configuration

% Initial conditions
T_0 = 0.0;                                              % dT at time = 0 [K]
N_0 = 0e22;                                             % dN at time = 0 [m^(-3)]
a_0 = 0;                                                % complex modal amplitude inside MRR at time = 0 [sqrt(J)]


%% Create input signal. 
% This function also returns the target series we want
% to predict, as well as the input signal used to modulate the laser light,
% as well as creating the target values.
[Ein, input, target] = create_input_signal_Github(total_L, steps_per_bit, ...
    N_mask, Pin_avg_lin, bias, type, seed, P);


%% 3D solver
% Solver for system with 3 differential equations. For reservoir
% computing, we only care about E_drop for further processing.
[E_drop, E_through, E_add, a_norm, theta_norm, n_norm, tau, a, delta_Temp, delta_N, delta_freq_vs_time] = RK4_4D_norm_Github(time_axis, Ein, T_0, N_0, a_0, solver_steps, tau_d, ...
    tau_th, tau_fc, tau_r, tau_c, eta_lin, omega_resonance_0, omega_p, m, cp, nsi, ...
    dnsi_dT, dnsi_dN, GAMMA_TH, GAMMA_FCA, GAMMA_TPA, V_FCA, V_TPA, B_TPA, c, h_bar, sigma_fca, zeta, phi_d, mode);

%% 2D solver. For some reason, the 3D solver provides much better NMSE when performing reservoir computing. 
% Solver for system with 2 differential equations. For reservoir computing, 
% we only care about E_drop_2D for further processing.
%[E_drop_2D, E_through_2D, E_add_2D, a_norm_2D, theta_norm_2D, n_norm_2D, tau_2D, a_2D, delta_Temp_2D, delta_N_2D] = RK4_2D_Github(time_axis, Ein, T_0, N_0, a_0, solver_steps, tau_d, ...
%    tau_th, tau_fc, tau_r, tau_c, eta_lin, omega_resonance_0, omega_p, m, cp, nsi, ...
%    dnsi_dT, dnsi_dN, GAMMA_TH, GAMMA_FCA, V_FCA, B_TPA, c, h_bar, sigma_fca, t_sq, k_sq, zeta, phi_d, mode);


%% Readout layer

% Use this line if you have used the 2D solver
%[P_drop, X_drop,time_axis_reduced, X_train, X_test, y_train, y_test, y_train_hat, y_test_hat] = readout_layer_Github(E_drop_2D, target, time_axis, steps_per_bit, N_mask, Warm1, Warm2, Ltrain, Ltest, total_L);

%Use this line if you have used the 4D solver
[P_drop, X_drop,time_axis_reduced, X_train, X_test, y_train, y_test, y_train_hat, y_test_hat] = readout_layer_Github(E_drop, target, time_axis, steps_per_bit, N_mask, Warm1, Warm2, Ltrain, Ltest, total_L);

% Calculate NMSE on both training and test set.
NMSE_train = calc_NMSE_Github(y_train, y_train_hat) 
NMSE_test = calc_NMSE_Github(y_test, y_test_hat)



%% Compare predicted values to true values
plot(y_test_hat(2:end-1)); hold on
plot(y_test(2:end-1));

legend({'$\mathbf{ \hat{y}}_{\mathrm{test}}$', '$\mathbf{{y}}_{\mathrm{test}}$'}, 'Interpreter', 'latex', 'FontSize', 14)

xlabel('Sample number', 'Interpreter', 'latex')
ylabel('Value', 'Interpreter', 'latex')

title(['$\mathrm{NARMA10},\; P_{\mathrm{in}} = ' num2str(Pin_avg_dBm,'%.0f') ...
       '\,\mathrm{dBm},\; \frac{\delta\omega}{2\pi} = ' num2str(delta_freq*1e-9,'%.0f') ...
       '\,\mathrm{GHz},\; \mathrm{NMSE} = ' num2str(NMSE_test,'%.4f') '$'], ...
      'Interpreter','latex','FontSize',14)


