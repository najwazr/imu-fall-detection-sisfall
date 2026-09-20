clc; clear; close all;

%% Load Data
fall_file = 'SisFall_data/SA01/F01_SA01_R01.txt';
data      = readmatrix(fall_file);

% Convert
accel    = data(:, 1:3) / 256;       % g
gyro_deg = data(:, 4:6) / 16.384;    % deg/s
gyro_rad = gyro_deg * (pi/180);      % rad/s

fs = 200;
dt = 1/fs;
t  = (0:length(accel)-1) / fs;

%% Before
svm = sqrt(sum(accel.^2, 2));

roll_raw  = cumsum(gyro_deg(:,1) * dt);
pitch_raw = cumsum(gyro_deg(:,2) * dt);

%% After
beta = 0.033;
q    = [1, 0, 0, 0];
orientation_filtered = zeros(length(accel), 2);

for k = 1:length(accel)
    q = madgwickIMU(q, gyro_rad(k,:), accel(k,:), beta, dt);
    qw=q(1); qx=q(2); qy=q(3); qz=q(4);
    roll_k  = atan2d(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2));
    pitch_k = asind(max(-1, min(1, 2*(qw*qy - qz*qx))));
    orientation_filtered(k,:) = [roll_k, pitch_k];
end

roll_filtered  = orientation_filtered(:,1);
pitch_filtered = orientation_filtered(:,2);

%% Plot
figure('Position', [100 100 1400 900]);

% === SVM ===
subplot(3,2,1);
plot(t, accel(:,1), 'b', t, accel(:,2), 'g', t, accel(:,3), 'r');
xlabel('Time (s)'); ylabel('Acceleration (g)');
title('Raw Accelerometer Signal (ax, ay, az)');
legend('ax','ay','az');
grid on;

subplot(3,2,2);
plot(t, svm, 'k', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('SVM (g)');
title('Signal Vector Magnitude (SVM)');
grid on;

% === Roll ===
subplot(3,2,3);
plot(t, roll_raw, 'b', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Roll (deg)');
title('Roll - Gyroscope Integration');
grid on;

subplot(3,2,4);
plot(t, roll_filtered, 'r', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Roll (deg)');
title('Roll - After Madgwick Filter');
grid on;

% === Pitch ===
subplot(3,2,5);
plot(t, pitch_raw, 'b', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Pitch (deg)');
title('Pitch - Gyroscope Integration');
grid on;

subplot(3,2,6);
plot(t, pitch_filtered, 'r', 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Pitch (deg)');
title('Pitch - After Madgwick Filter');
grid on;

%% =========================================================
function q = madgwickIMU(q, gyro, accel, beta, dt)
if norm(accel) == 0, return; end
accel = accel / norm(accel);

qw=q(1); qx=q(2); qy=q(3); qz=q(4);

F = [2*(qx*qz - qw*qy) - accel(1);
     2*(qw*qx + qy*qz) - accel(2);
     2*(0.5 - qx^2 - qy^2) - accel(3)];

J = [-2*qy,  2*qz, -2*qw, 2*qx;
      2*qx,  2*qw,  2*qz, 2*qy;
      0,    -4*qx, -4*qy, 0   ];

step = J' * F;
if norm(step) ~= 0
    step = step / norm(step);
else
    step = zeros(4,1);
end

qDot = 0.5 * quatmultiply(q, [0, gyro(1), gyro(2), gyro(3)]) - beta * step';
q    = q + qDot * dt;
q    = q / norm(q);
end

function result = quatmultiply(q, r)
result = [q(1)*r(1) - q(2)*r(2) - q(3)*r(3) - q(4)*r(4);
          q(1)*r(2) + q(2)*r(1) + q(3)*r(4) - q(4)*r(3);
          q(1)*r(3) - q(2)*r(4) + q(3)*r(1) + q(4)*r(2);
          q(1)*r(4) + q(2)*r(3) - q(3)*r(2) + q(4)*r(1)]';
end
