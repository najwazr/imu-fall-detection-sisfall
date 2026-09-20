%% PREPROCESSING
% 24 features | SisFall dataset | 200Hz | Window 1s | Overlap 50%
% Sucerquia et al. (2017) - SisFall: A Fall Detection Dataset

clc; clear; close all;
warning('off', 'all');

fprintf('=== PREPROCESSING FOR MACHINE LEARNING ===\n\n');

dataPath       = 'SisFall_data/';
features_table = table();
subjects       = dir(fullfile(dataPath, 'SA*'));
total_files    = 0;

fprintf('Found %d subjects\n', length(subjects));
fprintf('Starting feature extraction...\n\n');

%% Feature Extraction Loop
for s = 1:length(subjects)
    subjectFolder = fullfile(subjects(s).folder, subjects(s).name);
    files = dir(fullfile(subjectFolder, '*.txt'));
    subjectID = subjects(s).name;  % e.g. 'SA01'

    fprintf('Subject %d/%d: %s (%d files)\n', ...
        s, length(subjects), subjectID, length(files));

    for f = 1:length(files)
        if mod(f, 10) == 0
            fprintf('  Processing file %d/%d...\n', f, length(files));
        end

        try
            %% Load & Convert
            data     = readmatrix(fullfile(subjectFolder, files(f).name));
            % LSB -> g (ADXL345, +-16g, 256 LSB/g)
            accel    = data(:, 1:3) / 256;
            % LSB -> deg/s (ITG3200, 16.384 LSB/deg/s)
            gyro_deg = data(:, 4:6) / 16.384;
            % deg/s -> rad/s
            gyro_rad = gyro_deg * (pi/180);

            %% Madgwick Filter (IMU mode - no magnetometer)
            fs   = 200; dt = 1/fs; beta = 0.033;
            q    = [1, 0, 0, 0];
            orientation_euler = zeros(length(accel), 2);

            for k = 1:length(accel)
                q = madgwickIMU(q, gyro_rad(k,:), accel(k,:), beta, dt);
                qw = q(1); qx = q(2); qy = q(3); qz = q(4);
                roll_k  = atan2d(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2));
                % Clamp for numerical stability
                pitch_k = asind(max(-1, min(1, 2*(qw*qy - qz*qx))));
                orientation_euler(k,:) = [roll_k, pitch_k];
            end

            roll  = orientation_euler(:,1);
            pitch = orientation_euler(:,2);

            % Remove NaN samples if any
            validIdx = ~isnan(roll) & ~isnan(pitch);
            roll     = roll(validIdx);
            pitch    = pitch(validIdx);
            accel    = accel(validIdx,:);
            gyro_rad = gyro_rad(validIdx,:);

            %% Derived Signals
            svm      = sqrt(sum(accel.^2, 2));     % Signal Vector Magnitude (g)
            gyro_mag = sqrt(sum(gyro_rad.^2, 2));  % Gyroscope magnitude (rad/s)

            % Skip files that are too short
            windowSize = 200; overlap = 100;
            if length(svm) < windowSize, continue; end

            % File name without extension, e.g. 'F01_SA01_R01'
            [~, baseFileName, ~] = fileparts(files(f).name);

            %% Sliding Window
            for i = 1:overlap:(length(svm) - windowSize + 1)
                ws = svm(i:i+windowSize-1);              % SVM window
                wa = accel(i:i+windowSize-1, :);         % Accelerometer window
                wg = gyro_mag(i:i+windowSize-1);         % Gyroscope magnitude window
                wr = roll(i:i+windowSize-1);             % Roll window [-180, +180 deg] (atan2d)
                wp = pitch(i:i+windowSize-1);            % Pitch window [ -90,  +90 deg] (asind)

                %% === 24 FEATURES ===

                % --- Group 1: SVM Features (8 features) ---
                svm_min      = min(ws);
                svm_max      = max(ws);
                svm_mean     = mean(ws);
                svm_std      = std(ws);
                svm_range    = svm_max - svm_min;
                svm_rms      = rms(ws);
                svm_skewness = skewness(ws);
                svm_kurtosis = kurtosis(ws);

                % --- Group 2: Acceleration per Axis (6 features) ---
                accel_mean_x = mean(wa(:,1));
                accel_mean_y = mean(wa(:,2));
                accel_mean_z = mean(wa(:,3));
                accel_std_x  = std(wa(:,1));
                accel_std_y  = std(wa(:,2));
                accel_std_z  = std(wa(:,3));

                % --- Group 3: Peak Feature (1 feature) ---
                try
                    num_peaks = length(findpeaks(ws, 'MinPeakHeight', 1.5));
                catch
                    num_peaks = 0;
                end

                % --- Group 4: Orientation Features (4 features) ---
                roll_std    = std(wr);
                pitch_std   = std(wp);
                roll_range  = range(wr);
                pitch_range = range(wp);

                % --- Group 5: Gyroscope Features (2 features) ---
                gyro_std     = std(wg);
                gyro_mag_max = max(wg);

                % --- Group 6: Advanced Signal Features (3 features) ---
                energy   = sum(ws.^2) / windowSize;
                sma      = sum(abs(wa(:))) / windowSize;
                centered = ws - mean(ws);
                zcr      = sum(abs(diff(sign(centered)))) / 2;

                %% Label: F = Fall, D = ADL (Not_Fall)
                if startsWith(files(f).name, 'F')
                    label = 'Fall';
                else
                    label = 'Not_Fall';
                end

                windowStartSample = i;
                windowMaxSVM      = svm_max;  % Used as a proxy for impact intensity

                %% Append Row
                newRow = table(...
                    {subjectID}, {baseFileName}, windowStartSample, ...
                    svm_min, svm_max, svm_mean, svm_std, svm_range, ...
                    svm_rms, svm_skewness, svm_kurtosis, ...
                    accel_mean_x, accel_mean_y, accel_mean_z, ...
                    accel_std_x, accel_std_y, accel_std_z, ...
                    num_peaks, ...
                    roll_std, pitch_std, roll_range, pitch_range, ...
                    gyro_std, gyro_mag_max, ...
                    energy, sma, zcr, ...
                    {label}, ...
                    'VariableNames', {...
                        'SubjectID', 'FileName', 'WindowStartSample', ...
                        'SVM_Min', 'SVM_Max', 'SVM_Mean', 'SVM_Std', 'SVM_Range', ...
                        'SVM_RMS', 'SVM_Skewness', 'SVM_Kurtosis', ...
                        'Accel_Mean_X', 'Accel_Mean_Y', 'Accel_Mean_Z', ...
                        'Accel_Std_X', 'Accel_Std_Y', 'Accel_Std_Z', ...
                        'Num_Peaks', ...
                        'Roll_Std', 'Pitch_Std', 'Roll_Range', 'Pitch_Range', ...
                        'Gyro_Std', 'Gyro_Mag_Max', ...
                        'Energy', 'SMA', 'ZCR', ...
                        'Label'});
                features_table = [features_table; newRow];
            end
            total_files = total_files + 1;

        catch ME
            fprintf('  Warning: %s - %s\n', files(f).name, ME.message);
        end
    end
end

warning('on', 'all');

%% Convert & Save
features_table.Label     = categorical(features_table.Label);
features_table.SubjectID = categorical(features_table.SubjectID);
save('features_dataset_v3.mat', 'features_table');
writetable(features_table, 'features_dataset_v3.csv');

%% Summary & Sanity Check
fprintf('\n=== PREPROCESSING COMPLETE ===\n\n');
fprintf('Dataset Statistics:\n');
fprintf('  Total files processed : %d\n', total_files);
fprintf('  Total windows         : %d\n', height(features_table));
fprintf('  Total subjects        : %d\n', numel(categories(features_table.SubjectID)));
fprintf('  Fall windows          : %d (%.1f%%)\n', ...
    sum(features_table.Label == 'Fall'), ...
    sum(features_table.Label == 'Fall') / height(features_table) * 100);
fprintf('  Not_Fall windows      : %d (%.1f%%)\n', ...
    sum(features_table.Label == 'Not_Fall'), ...
    sum(features_table.Label == 'Not_Fall') / height(features_table) * 100);

fprintf('\nFeature Ranges (sanity check):\n');
fprintf('  SVM_Min      : %.4f - %.4f g\n',      min(features_table.SVM_Min),      max(features_table.SVM_Min));
fprintf('  SVM_Max      : %.4f - %.4f g\n',      min(features_table.SVM_Max),      max(features_table.SVM_Max));
fprintf('  SVM_Skewness : %.4f - %.4f\n',        min(features_table.SVM_Skewness), max(features_table.SVM_Skewness));
fprintf('  SVM_Kurtosis : %.4f - %.4f\n',        min(features_table.SVM_Kurtosis), max(features_table.SVM_Kurtosis));
fprintf('  Gyro_Std     : %.4f - %.4f rad/s\n',  min(features_table.Gyro_Std),     max(features_table.Gyro_Std));
fprintf('  Gyro_Mag_Max : %.4f - %.4f rad/s\n',  min(features_table.Gyro_Mag_Max), max(features_table.Gyro_Mag_Max));
fprintf('  Roll_Range   : %.2f  - %.2f deg\n',   min(features_table.Roll_Range),   max(features_table.Roll_Range));
fprintf('  Pitch_Range  : %.2f  - %.2f deg\n',   min(features_table.Pitch_Range),  max(features_table.Pitch_Range));
fprintf('  NaN count    : %d\n', sum(sum(ismissing(features_table))));

if max(features_table.SVM_Max) > 100
    fprintf('\n  WARNING: SVM_Max is too high, check unit conversion!\n');
else
    fprintf('\n  Values look reasonable!\n');
end

fprintf('\nReady for ML training!\n');
fprintf('Saved: features_dataset_v3.mat & .csv\n');
fprintf('Extra metadata columns: SubjectID, FileName, WindowStartSample\n');

%% =========================================================
%  LOCAL FUNCTIONS
%% =========================================================

function q = madgwickIMU(q, gyro, accel, beta, dt)
% Madgwick IMU filter (no magnetometer)
% Input : q     - current quaternion [qw qx qy qz]
%         gyro  - angular velocity (rad/s) [wx wy wz]
%         accel - acceleration (g) [ax ay az]
%         beta  - accelerometer correction gain
%         dt    - time step (s)
% Output: q     - updated quaternion

% Normalize accelerometer
if norm(accel) == 0, return; end
accel = accel / norm(accel);

% Extract quaternion components
qw = q(1); qx = q(2); qy = q(3); qz = q(4);

% Objective function
F = [2*(qx*qz - qw*qy) - accel(1);
     2*(qw*qx + qy*qz) - accel(2);
     2*(0.5 - qx^2 - qy^2) - accel(3)];
 
% Jacobian
J = [-2*qy,  2*qz, -2*qw, 2*qx;
      2*qx,  2*qw,  2*qz, 2*qy;
      0,    -4*qx, -4*qy, 0   ];

% Gradient descent step
step = J' * F;
if norm(step) ~= 0
    step = step / norm(step);
else
    step = zeros(4,1);
end

% Update quaternion
qDot = 0.5 * quatMultiply(q, [0, gyro(1), gyro(2), gyro(3)]) - beta * step';
q    = q + qDot * dt;
q    = q / norm(q);
end

function result = quatMultiply(q, r)
result = [q(1)*r(1) - q(2)*r(2) - q(3)*r(3) - q(4)*r(4);
          q(1)*r(2) + q(2)*r(1) + q(3)*r(4) - q(4)*r(3);
          q(1)*r(3) - q(2)*r(4) + q(3)*r(1) + q(4)*r(2);
          q(1)*r(4) + q(2)*r(3) - q(3)*r(2) + q(4)*r(1)]';
end
