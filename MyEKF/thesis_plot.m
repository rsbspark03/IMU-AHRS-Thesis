clear;
clc;

% --- 1. Load the .mat file ---
% Make sure '30z.mat' is in MATLAB's current working directory or on the path.
load('30z.mat');
col = 3;
% Verify that the variables are loaded (optional, for debugging)
% who; % This command lists all variables in the workspace

% --- 2. Extract the Z-column data ---
% Assuming 'My_EKF_Data' and 'Onboard_Filter_Data' are Nx3 arrays.
% The Z-column is the 3rd column (index 3).
ekf_z_data = My_EKF(:, col);
onboard_z_data = Onboard_Filter(:, col);

% --- 3. Create the plot ---
figure; % Opens a new figure window

% Plot My EKF Z-column
plot(Time, ekf_z_data, 'b-', 'DisplayName', 'My EKF (Z-axis)');
hold on; % Keep the current plot to add more lines

% Plot Onboard Filter Z-column
plot(Time, onboard_z_data, 'r--', 'DisplayName', 'Onboard Filter (Z-axis)');

% --- 4. Draw reference lines ---
% Get the current x-axis limits to extend the reference lines across the plot
x_limits = xlim;

% Draw y=30 reference line
plot(x_limits, [30 30], 'k:', 'LineWidth', 1, 'DisplayName', 'Ref: 30');

% Draw y=0 reference line
plot(x_limits, [0 0], 'g:', 'LineWidth', 1, 'DisplayName', 'Ref: 0');


% --- 5. Add plot enhancements ---
title('Z-axis Orientation Comparison: My EKF vs. Onboard Filter');
xlabel('Time (s)');
ylabel('Angle (Degrees)');
legend('Location', 'best'); % Place legend where it fits best
grid on; % Add a grid for better readability
hold off; % Release the plot so new plots open in a new figure (optional)