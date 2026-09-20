%% ABLATION MADGWICK

clc; clear; close all;
fprintf('=== ABLATION MADGWICK: 20 FEATURES ===\n\n');

%% Load Dataset
load('features_dataset_v3.mat');

% Exclude the 4 Madgwick orientation features
featureNames = {'SVM_Min','SVM_Max','SVM_Mean','SVM_Std','SVM_Range', ...
    'SVM_RMS','SVM_Skewness','SVM_Kurtosis', ...
    'Accel_Mean_X','Accel_Mean_Y','Accel_Mean_Z', ...
    'Accel_Std_X','Accel_Std_Y','Accel_Std_Z', ...
    'Num_Peaks', ...
    'Gyro_Std','Gyro_Mag_Max', ...
    'Energy','SMA','ZCR'};

X = features_table{:, featureNames};
y = features_table.Label;
subjectID = features_table.SubjectID;
uniqueSubjects = categories(subjectID);
nSubjects = numel(uniqueSubjects);

fprintf('Total windows : %d\n', height(features_table));
fprintf('Total subjects: %d\n', nSubjects);
fprintf('Total features: %d (without orientation features)\n\n', numel(featureNames));

%% Subject-Wise Partition
rng(42);
subjectFoldAssignment = crossvalind('Kfold', nSubjects, 10);
subjectToFold = containers.Map(uniqueSubjects, num2cell(subjectFoldAssignment));
windowFold = zeros(height(features_table), 1);

for i = 1:height(features_table)
    windowFold(i) = subjectToFold(char(subjectID(i)));
end

fprintf('=== SUBJECT-WISE PARTITION VERIFICATION ===\n');
for foldNum = 1:10
    subj = unique(subjectID(windowFold == foldNum));
    fprintf('  Fold %2d: %s\n', foldNum, strjoin(cellstr(subj), ','));
end
fprintf('\n');

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
                [pred, scores] = predict(mdl, Xtest);
                
            case 'RF_SelectAll'
                template = templateTree('MaxNumSplits', 116496, 'NumVariablesToSample', 'all');
                mdl = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', ...
                    'NumLearningCycles', 100, 'Learners', template);
                [pred, scores] = predict(mdl, Xtest);
                
            case 'RF_SqrtP'
                % NumVariablesToSample kept at 5 for consistency with the 24-feature run
                template = templateTree('MaxNumSplits', 116496, 'NumVariablesToSample', 5);
                mdl = fitcensemble(Xtrain, ytrain, 'Method', 'Bag', ...
                    'NumLearningCycles', 100, 'Learners', template);
                [pred, scores] = predict(mdl, Xtest);
                
            case 'KNN_Fine'
                mdl = fitcknn(Xtrain, ytrain, 'Distance', 'Euclidean', 'Exponent', [], ...
                    'NumNeighbors', 1, 'DistanceWeight', 'Equal', 'Standardize', true);
                [pred, scores] = predict(mdl, Xtest);
                
            case 'KNN_Weighted'
                mdl = fitcknn(Xtrain, ytrain, 'Distance', 'Euclidean', 'Exponent', [], ...
                    'NumNeighbors', 10, 'DistanceWeight', 'SquaredInverse', 'Standardize', true);
                [pred, scores] = predict(mdl, Xtest);
                
            case 'SVM_Quadratic'
                mdl = fitcsvm(Xtrain, ytrain, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', 2, 'KernelScale', 'auto', 'BoxConstraint', 1, ...
                    'Standardize', true);
                [pred, scores] = predict(mdl, Xtest);
                
            case 'SVM_Cubic'
                mdl = fitcsvm(Xtrain, ytrain, 'KernelFunction', 'polynomial', ...
                    'PolynomialOrder', 3, 'KernelScale', 'auto', 'BoxConstraint', 1, ...
                    'Standardize', true);
                [pred, scores] = predict(mdl, Xtest);
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
        
        fprintf('  %-14s TP=%5d FN=%5d FP=%5d TN=%5d | Acc=%.1f%% | %.1fs\n', ...
            modelName, TP, FN, FP, TN, acc*100, elapsedModel);
        
        if foldNum == foldsToRun(end)
            classNames = mdl.ClassNames;
            fallIdx = find(classNames == 'Fall');
            [~,~,~, AUC] = perfcurve(ytest, scores(:, fallIdx), 'Fall');
            fprintf('  %-14s AUC (fold %d) = %.4f\n', modelName, foldNum, AUC);
        end

    end
    elapsedFold = toc(foldTimerStart);
    fprintf('  --- Fold %d finished in %.1f minutes ---\n\n', foldNum, elapsedFold/60);
    
    if QUICK_TEST
        estimatedTotal = elapsedFold * 10 / 60;
        fprintf('*** ESTIMATED FULL RUN TIME (10 folds): %.1f minutes ***\n\n', estimatedTotal);
    end
end

%% Summary & Save
totalElapsed = toc(totalTimerStart);
fprintf('=== ABLATION FINAL RESULTS ===\n');
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
    
    writetable(T, sprintf('perfold_detail_ABLATION_%s.csv', modelName));
end

writetable(summaryTable, 'hasil_ablation_20fitur_summary.csv');
fprintf('\nSaved: hasil_ablation_20fitur_summary.csv & per-model detail .csv\n');
