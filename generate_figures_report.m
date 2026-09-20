%% GENERATE FIGURES: CONFUSION MATRIX & ROC (LAST FOLD)

clc; clear; close all;

%% Load Models
load('trained_models_lastfold.mat');  % -> savedModels

modelNames = {'DT_FineTree', 'RF_SelectAll', 'RF_SqrtP', ...
              'KNN_Fine', 'KNN_Weighted', 'SVM_Quadratic', 'SVM_Cubic'};

if ~exist('figures_report', 'dir'), mkdir('figures_report'); end

%% Generate Figures
for m = 1:numel(modelNames)
    modelName = modelNames{m};
    label = strrep(modelName, '_', ' ');
    mdl   = savedModels.(modelName).model;
    Xtest = savedModels.(modelName).Xtest;
    ytest = savedModels.(modelName).ytest;

    [pred, scores] = predict(mdl, Xtest);

    % Confusion matrix (counts)
    fig = figure('Visible', 'off', 'Position', [100 100 600 500]);
    cm = confusionchart(ytest, pred, ...
        'Title', sprintf('Confusion Matrix (Count) - %s', label));
    cm.FontSize = 12;
    saveas(fig, fullfile('figures_report', sprintf('cm_count_%s.png', modelName)));
    close(fig);

    % Confusion matrix (TPR/FNR, row-normalized)
    fig = figure('Visible', 'off', 'Position', [100 100 600 500]);
    cm = confusionchart(ytest, pred, ...
        'Title', sprintf('Confusion Matrix (TPR/FNR) - %s', label), ...
        'RowSummary', 'row-normalized', 'Normalization', 'row-normalized');
    cm.FontSize = 12;
    saveas(fig, fullfile('figures_report', sprintf('cm_tprfnr_%s.png', modelName)));
    close(fig);

    % ROC (positive class = Fall)
    fallIdx = find(mdl.ClassNames == 'Fall');
    [fpr, tpr, ~, AUC] = perfcurve(ytest, scores(:, fallIdx), 'Fall');

    fig = figure('Visible', 'off', 'Position', [100 100 600 500]);
    plot(fpr, tpr, 'LineWidth', 2, 'Color', [0 0.4470 0.7410]);
    hold on;
    plot([0 1], [0 1], '--', 'Color', [0.5 0.5 0.5]);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve - %s (AUC = %.4f)', label, AUC));
    legend(sprintf('%s (AUC=%.3f)', label, AUC), 'Random Classifier', 'Location', 'southeast');
    grid on;
    saveas(fig, fullfile('figures_report', sprintf('roc_%s.png', modelName)));
    close(fig);

    fprintf('%-14s: AUC = %.4f\n', modelName, AUC);
end

fprintf('\nSaved: figures_report/ (cm_count_, cm_tprfnr_, roc_ per model)\n');
