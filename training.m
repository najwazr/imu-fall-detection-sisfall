%% SUBJECT-WISE CROSS-VALIDATION: 24 FEATURES
% Trains 7 models using subject-wise 10-fold CV, ensuring windows from
% the same subject never appear in both train and test sets within a fold.
%
% Model hyperparameters were verified directly against MATLAB's
% Classification Learner App "Generate Function" output, not assumed.

clc; clear; close all;
fprintf('=== SUBJECT-WISE 10-FOLD CROSS-VALIDATION: 24 FEATURES ===\n\n');

%% Load Dataset
load('features_dataset_v3.mat');

featureNames = {'SVM_Min','SVM_Max','SVM_Mean','SVM_Std','SVM_Range', ...
    'SVM_RMS','SVM_Skewness','SVM_Kurtosis', ...
    'Accel_Mean_X','Accel_Mean_Y','Accel_Mean_Z', ...
    'Accel_Std_X','Accel_Std_Y','Accel_Std_Z', ...
    'Num_Peaks', ...
    'Roll_Std','Pitch_Std','Roll_Range','Pitch_Range', ...
    'Gyro_Std','Gyro_Mag_Max', ...
    'Energy','SMA','ZCR'};

X = features_table{:, featureNames};
y = features_table.Label;
subjectID = features_table.SubjectID;
uniqueSubjects = categories(subjectID);
nSubjects = numel(uniqueSubjects);

fprintf('Total windows : %d\n', height(features_table));
fprintf('Total subjects: %d\n', nSubjects);
fprintf('Total features: %d\n\n', numel(featureNames));

%% Subject-Wise Partition
rng(42);
subjectFoldAssignment = crossvalind('Kfold', nSubjects, 10);
subjectToFold = containers.Map(uniqueSubjects, num2cell(subjectFoldAssignment));
windowFold = zeros(height(features_table), 1);

for i = 1:height(features_table)
    windowFold(i) = subjectToFold(char(subjectID(i)));
end

%% Partition Verification (leakage check)
fprintf('=== SUBJECT-WISE PARTITION VERIFICATION ===\n');
for foldNum = 1:10
    inFold  = unique(subjectID(windowFold == foldNum));
    outFold = unique(subjectID(windowFold ~= foldNum));
    if ~isempty(intersect(inFold, outFold))
        error('Leakage detected in fold %d!', foldNum);
    end
    fprintf('  Fold %2d: %s (%d subjects, %d windows)\n', foldNum, ...
        strjoin(cellstr(inFold), ','), numel(inFold), sum(windowFold == foldNum));
end
fprintf('  OK: no subject overlaps across folds.\n\n');

%% Model Definitions
modelNames = {'DT_FineTree', 'RF_SelectAll', 'RF_SqrtP', ...
              'KNN_Fine', 'KNN_Weighted', 'SVM_Quadratic', 'SVM_Cubic'};
nModels = numel(modelNames);
perFoldMetrics = struct();

for m = 1:nModels
    perFoldMetrics.(modelNames{m}) = table();
end

QUICK_TEST = false;
if QUICK_TEST
    foldsToRun = 1;
    fprintf('*** QUICK TEST MODE: Fold 1 only ***\n\n');
else
    foldsToRun = 1:10;
end

totalTimerStart = tic;

%% Cross-Validation Loop
for foldNum = foldsToRun
    fprintf('=== FOLD %d/10 ===\n', foldNum);
    foldTimerStart = tic;

    testIdx  = (windowFold == foldNum);
    trainIdx = ~testIdx;
    Xtrain = X(trainIdx, :);  ytrain = y(trainIdx);
    Xtest  = X(testIdx, :);   ytest  = y(testIdx);

    for m = 1:nModels
        modelName = modelNames{m};
        modelTimerStart = tic;

        switch modelName
            case 'DT_FineTree'
                mdl = fitctree(Xtrain, ytrain, ...
                    'SplitCriterion', 'gdi', 'MaxNumSplits', 100, 'Surrogate', 'off');

            case 'RF_SelectAll'
                template = templateTree('MaxNumSplits', 116496, 'NumVariablesToSample', 'all');
                mdl = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', ...
                    'NumLearningCycles', 100, 'Learners', template);

            case 'RF_SqrtP'
                template = templateTree('MaxNumSplits', 116496, 'NumVariablesToSample', 5);
                mdl = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', ...
                    'NumLearningCycles', 100, 'Learners', template);

            case 'KNN_Fine'
                mdl = fitcknn(Xtrain, ytrain, 'Distance', 'Euclidean', 'Exponent', [], ...
                    'NumNeighbors', 1, 'DistanceWeight', 'Equal', 'Standardize', true);

            case 'KNN_Weighted'
                mdl = fitcknn(Xtrain, ytrain, 'Distance', 'Euclidean', 'Exponent', [], ...
                    'NumNeighbors', 10, 'DistanceWeight', 'SquaredInverse', 'Standardize', true);

            case 'SVM_Quadratic'
                mdl = fitcsvm(Xtrain, ytrain, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', 2, 'KernelScale', 'auto', 'BoxConstraint', 1, ...
                    'Standardize', true);

            case 'SVM_Cubic'
                % IterationLimit is left at default (a 5M limit made training exceed 1 hour)
                mdl = fitcsvm(Xtrain, ytrain, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', 3, 'KernelScale', 'auto', 'BoxConstraint', 1, ...
                    'Standardize', true);
        end

        pred = predict(mdl, Xtest);

        % Check SVM convergence
        if startsWith(modelName, 'SVM') && ~mdl.ConvergenceInfo.Converged
            fprintf('    WARNING: %s did not converge on fold %d (%s)\n', ...
                modelName, foldNum, mdl.ConvergenceInfo.ReasonForConvergence);
        end

        elapsedModel = toc(modelTimerStart);

        % Compute Metrics
        TP = sum(pred == 'Fall'     & ytest == 'Fall');
        FN = sum(pred == 'Not_Fall' & ytest == 'Fall');
        FP = sum(pred == 'Fall'     & ytest == 'Not_Fall');
        TN = sum(pred == 'Not_Fall' & ytest == 'Not_Fall');

        acc  = (TP+TN)/(TP+TN+FP+FN);
        sens = TP/(TP+FN);
        spec = TN/(TN+FP);
        prec = TP/(TP+FP);
        f1   = 2*prec*sens/(prec+sens);

        newFoldRow = table(foldNum, TP, FN, FP, TN, acc, sens, spec, prec, f1, elapsedModel, ...
            'VariableNames', {'Fold','TP','FN','FP','TN','Accuracy','Sensitivity', ...
            'Specificity','Precision','F1Score','TrainPredictTime_sec'});
        perFoldMetrics.(modelName) = [perFoldMetrics.(modelName); newFoldRow];

        % Save last-fold model & test data (used by generate_figures_report.m)
        if foldNum == foldsToRun(end)
            savedModels.(modelName).model = mdl;
            savedModels.(modelName).Xtest = Xtest;
            savedModels.(modelName).ytest = ytest;
        end

        fprintf('  %-14s TP=%5d FN=%5d FP=%5d TN=%5d | Acc=%.1f%% | %.1fs\n', ...
            modelName, TP, FN, FP, TN, acc*100, elapsedModel);
    end

    elapsedFold = toc(foldTimerStart);
    fprintf('  --- Fold %d finished in %.1f minutes ---\n\n', foldNum, elapsedFold/60);

    if QUICK_TEST
        fprintf('*** ESTIMATED FULL RUN TIME (10 folds): %.1f minutes ***\n\n', elapsedFold*10/60);
    end
end

%% Summary & Save
totalElapsed = toc(totalTimerStart);
fprintf('=== FINAL RESULTS: SUBJECT-WISE CV ===\n');
fprintf('Total computation time: %.1f minutes\n\n', totalElapsed/60);
fprintf('%-14s %14s %14s %14s %14s %14s\n', 'Model', 'Acc(%)', 'Sens(%)', 'Spec(%)', 'Prec(%)', 'F1(%)');

summaryTable = table();
for m = 1:nModels
    modelName = modelNames{m};
    T = perFoldMetrics.(modelName);

    accMean  = mean(T.Accuracy)*100;    accSD  = std(T.Accuracy)*100;
    sensMean = mean(T.Sensitivity)*100; sensSD = std(T.Sensitivity)*100;
    specMean = mean(T.Specificity)*100; specSD = std(T.Specificity)*100;
    precMean = mean(T.Precision)*100;   precSD = std(T.Precision)*100;
    f1Mean   = mean(T.F1Score)*100;     f1SD   = std(T.F1Score)*100;

    fprintf('%-14s %6.2f+-%4.2f %6.2f+-%4.2f %6.2f+-%4.2f %6.2f+-%4.2f %6.2f+-%4.2f\n', ...
        modelName, accMean, accSD, sensMean, sensSD, specMean, specSD, ...
        precMean, precSD, f1Mean, f1SD);

    newRow = table({modelName}, accMean, accSD, sensMean, sensSD, ...
        specMean, specSD, precMean, precSD, f1Mean, f1SD, ...
        'VariableNames', {'Model','Accuracy_Mean','Accuracy_SD','Sensitivity_Mean', ...
        'Sensitivity_SD','Specificity_Mean','Specificity_SD','Precision_Mean', ...
        'Precision_SD','F1_Mean','F1_SD'});
    summaryTable = [summaryTable; newRow];

    writetable(T, sprintf('perfold_detail_%s.csv', modelName));
end

writetable(summaryTable, 'hasil_subjectwise_cv_summary.csv');
save('trained_models_lastfold.mat', 'savedModels');
fprintf('\nSaved: hasil_subjectwise_cv_summary.csv, per-model detail .csv, trained_models_lastfold.mat\n');
